#!/usr/bin/env python3

import rospy
import os
import numpy as np
import torch
import message_filters
import cv_bridge
from pathlib import Path
import open3d as o3d
import tf2_ros
from tf.transformations import quaternion_matrix
import cv2
import matplotlib.pyplot as plt
import time

from utils import *
from adet.config import get_cfg
from adet.utils.visualizer import visualize_pred_amoda_occ
from adet.utils.post_process import detector_postprocess, DefaultPredictor
from foreground_segmentation.model import Context_Guided_Network

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from grasp_msg.srv import GraspSegment, GraspSegmentResponse
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest
from uoais.msg import UOAISResults
from uoais.srv import UOAISRequest, UOAISRequestResponse


class UOAIS():

    def __init__(self):

        rospy.init_node("uoais")

        self.mode = rospy.get_param("~mode", "topic") 
        rospy.loginfo("Starting uoais node with {} mode".format(self.mode))
        self.rgb_topic = rospy.get_param("~rgb", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth", "/camera/aligned_depth_to_color/image_raw")
        # self.depth_topic = rospy.get_param("~depth", "/camera/depth/image_rect_raw")
        camera_info_topic = rospy.get_param("~camera_info", "/camera/color/camera_info")
        # UOAIS-Net
        self.det2_config_file = rospy.get_param("~config_file", 
                            "configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml")
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
        # CG-Net foreground segmentation
        self.use_cgnet = rospy.get_param("~use_cgnet", False)
        self.cgnet_weight = rospy.get_param("~cgnet_weight",
                             "foreground_segmentation/rgbd_fg.pth")
        # RANSAC plane segmentation
        self.use_planeseg = rospy.get_param("~use_planeseg", False)
        self.ransac_threshold = rospy.get_param("~ransac_threshold", 0.003)
        self.ransac_n = rospy.get_param("~ransac_n", 3)
        self.ransac_iter = rospy.get_param("~ransac_iter", 10)

        self.cv_bridge = cv_bridge.CvBridge()
        self.count = 0
        self.vis = False
        # initialize UOAIS-Net and CG-Net
        self.load_models()
        """
        add by matt, do some tf2 work to transform frame
        """
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)  # Create a tf listener
        self.target_center = None  # x, y and z center of target point cloud
        self.target_pointcloud = None  # point cloud of target
        self.o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                                        640, 480,
                                        617.0441284179688*640/640,
                                        617.0698852539062*480/480,
                                        322.3338317871094, 238.7687225341797)


        """
        end of add
        """

        if self.use_planeseg:
            camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
            self.K = camera_info.K
            self.o3d_camera_intrinsic = None

        if self.mode == "topic":
            rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
            depth_sub = message_filters.Subscriber(self.depth_topic, Image)
            self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=1, slop=2)
            self.ts.registerCallback(self.topic_callback)
            self.result_pub = rospy.Publisher("/uoais/results", UOAISResults, queue_size=10)
            rospy.loginfo("uoais results at topic: /uoais/results")

        elif self.mode == "service":
            # self.srv = rospy.Service('/get_uoais_results', UOAISRequest, self.service_callback)
            self.srv = rospy.Service('/get_uoais_results', GraspSegment, self.service_callback)
            self.points_pub = rospy.Publisher("/uoais/Pointclouds", PointCloud2, queue_size=10)
            rospy.loginfo("uoais results at service: /get_uoais_results")
        else:
            raise NotImplementedError
        self.vis_pub = rospy.Publisher("/uoais/vis_img", Image, queue_size=10)
        rospy.loginfo("visualization results at topic: /uoais/vis_img")

    def load_models(self):
        # UOAIS-Net
        self.det2_config_file = os.path.join(Path(__file__).parent.parent, self.det2_config_file)
        rospy.loginfo("Loading UOAIS-Net with config_file: {}".format(self.det2_config_file))
        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.det2_config_file)
        self.cfg.defrost()
        self.cfg.MODEL.WEIGHTS = os.path.join(Path(__file__).parent.parent, self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.confidence_threshold
        self.predictor = DefaultPredictor(self.cfg)
        self.W, self.H = self.cfg.INPUT.IMG_SIZE

        # CG-Net (foreground segmentation)
        if self.use_cgnet:
            checkpoint = torch.load(os.path.join(Path(__file__).parent.parent, self.cgnet_weight))
            self.fg_model = Context_Guided_Network(classes=2, in_channel=4)
            self.fg_model.load_state_dict(checkpoint['model'])
            self.fg_model.cuda()
            self.fg_model.eval()

    def topic_callback(self, rgb_msg, depth_msg):

        results = self.inference(rgb_msg, depth_msg)        
        self.result_pub.publish(results)

    def service_callback(self, msg):

        rgb_msg = rospy.wait_for_message(self.rgb_topic, Image)
        depth_msg = rospy.wait_for_message(self.depth_topic, Image)
        results = self.inference(rgb_msg, depth_msg)
        # return UOAISRequestResponse(results)
        return GraspSegmentResponse(results)

    def inference(self, rgb_msg, depth_msg):
        start_time = time.time()
        rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        ori_H, ori_W, _ = rgb_img.shape
        rgb_img = cv2.resize(rgb_img, (self.W, self.H))
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        depth_img = normalize_depth(depth)
        depth_img = cv2.resize(depth_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        self.count += 1
        print("0 time: {}".format(time.time() - start_time))
        # UOAIS-Net inference
        if self.cfg.INPUT.DEPTH and self.cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif self.cfg.INPUT.DEPTH and not self.cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)        
        outputs = self.predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], self.H, self.W).to('cuda')
        preds = instances.pred_masks.detach().cpu().numpy() 
        pred_visibles = instances.pred_visible_masks.detach().cpu().numpy() 
        pred_bboxes = instances.pred_boxes.tensor.detach().cpu().numpy() 
        pred_occs = instances.pred_occlusions.detach().cpu().numpy()
        print("1 time: {}".format(time.time() - start_time))
        # filter out the background instances 
        # CG-Net
        if self.use_cgnet:
            rospy.loginfo_once("Using foreground segmentation model (CG-Net) to filter out background instances")
            fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
            fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
            fg_depth_input = cv2.resize(depth_img, (320, 240)) 
            fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
            fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
            fg_output = self.fg_model(fg_input.cuda())
            fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
            fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
            fg_output = cv2.resize(fg_output, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        # RANSAC
        if self.use_planeseg:
            rospy.loginfo_once("Using RANSAC plane segmentation to filter out background instances")
            o3d_rgb_img = o3d.geometry.Image(rgb_img)
            o3d_depth_img = o3d.geometry.Image(unnormalize_depth(depth_img))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb_img, o3d_depth_img)
            if self.o3d_camera_intrinsic is None:
                self.o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                                            self.W, self.H, 
                                            self.K[0]*self.W/ori_W, 
                                            self.K[4]*self.H/ori_H, 
                                            self.K[2], self.K[5])
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(
                                                rgbd_image, self.o3d_camera_intrinsic)
            plane_model, inliers = o3d_pc.segment_plane(distance_threshold=self.ransac_threshold,
                                                        ransac_n=self.ransac_n,
                                                        num_iterations=self.ransac_iter)
            fg_output = np.ones(self.H * self.W)
            fg_output[inliers] = 0 
            fg_output = np.resize(fg_output, (self.H, self.W))
            fg_output = np.uint8(fg_output)

        if self.use_cgnet or self.use_planeseg:
            remove_idxs = []
            for i, pred_visible in enumerate(pred_visibles):
                iou = np.sum(np.bitwise_and(pred_visible, fg_output)) / np.sum(pred_visible)
                if iou < 0.5: 
                    remove_idxs.append(i)
            preds = np.delete(preds, remove_idxs, 0)
            pred_visibles = np.delete(pred_visibles, remove_idxs, 0)
            pred_bboxes = np.delete(pred_bboxes, remove_idxs, 0)
            pred_occs = np.delete(pred_occs, remove_idxs, 0)
        
        # reorder predictions for visualization
        idx_shuf = np.concatenate((np.where(pred_occs==1)[0] , np.where(pred_occs==0)[0] )) 
        preds, pred_visibles, pred_occs, pred_bboxes = \
            preds[idx_shuf], pred_visibles[idx_shuf], pred_occs[idx_shuf], pred_bboxes[idx_shuf]
        vis_img = visualize_pred_amoda_occ(rgb_img, preds, pred_bboxes, pred_occs)
        if self.use_cgnet or self.use_planeseg:
            vis_fg = np.zeros_like(rgb_img) 
            vis_fg[:, :, 1] = fg_output * 255
            vis_img = cv2.addWeighted(vis_img, 0.8, vis_fg, 0.2, 0)
        self.vis_pub.publish(self.cv_bridge.cv2_to_imgmsg(vis_img, encoding="bgr8"))
        # pubish the uoais results
        results = UOAISResults()
        # results.header = rgb_msg.header
        # results.bboxes = []
        # results.visible_masks = []
        # results.amodal_masks = []
        # n_instances = len(pred_occs)
        # for i in range(n_instances):
        #     bbox = RegionOfInterest()
        #     bbox.x_offset = int(pred_bboxes[i][0])
        #     bbox.y_offset = int(pred_bboxes[i][1])
        #     bbox.width = int(pred_bboxes[i][2]-pred_bboxes[i][0])
        #     bbox.height = int(pred_bboxes[i][3]-pred_bboxes[i][1])
        #     results.bboxes.append(bbox)
        #     results.visible_masks.append(self.cv_bridge.cv2_to_imgmsg(
        #                                 np.uint8(pred_visibles[i]), encoding="mono8"))
        #     results.amodal_masks.append(self.cv_bridge.cv2_to_imgmsg(
        #                                 np.uint8(preds[i]), encoding="mono8"))
        # results.occlusions = pred_occs.tolist()
        # results.class_names = ["object"] * n_instances
        # results.class_ids = [0] * n_instances
        print("2 time: {}".format(time.time() - start_time))
        """
        Extra part add by Matt, take depth image and mask as input to generate masked depth image
        and then turn it into pointcloud
        """
        try:
            """
            convert camera_color_optical_frame to camer_link
            """
            transform_stamped = self.tf_buffer.lookup_transform('base', depth_msg.header.frame_id, rospy.Time(0))
            trans = np.array([transform_stamped.transform.translation.x,
                              transform_stamped.transform.translation.y,
                              transform_stamped.transform.translation.z])
            quat = np.array([transform_stamped.transform.rotation.x,
                            transform_stamped.transform.rotation.y,
                            transform_stamped.transform.rotation.z,
                            transform_stamped.transform.rotation.w])
            T = quaternion_matrix(quat)
            T[:3, 3] = trans
            T_inv = np.linalg.inv(T)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to transform point cloud: {}".format(e))
            return

        print("3 time: {}".format(time.time() - start_time))
        print("T: {}".format(T))
        if self.target_pointcloud is None:
            np.set_printoptions(threshold=np.inf)
            ave_depth_list = []
            depth_list = []
            mask_center_list = []
            for i in range(len(pred_visibles)):
                mask = np.array(pred_visibles[i]).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)

                # choose by center
                indices = np.where(mask == 1)
                center_index = np.round(np.mean(indices, axis=1)).astype(int)
                print("center_index: {}".format(center_index))
                # mask_center_list.append(np.linalg.norm(center_index - np.array([240, 320])))
                # cv2.imshow('Image', mask.astype(np.uint8)*255)
                # # Wait for a key press to exit
                # cv2.waitKey(0)
                # # Close all windows
                # cv2.destroyAllWindows()

                un_depth_img = unnormalize_depth(depth_img)
                un_depth_img[np.logical_not(mask)] = 0
                # choose by depth
                ave_depth_list.append(np.mean(un_depth_img[un_depth_img != 0]))

                depth_list.append(un_depth_img)
            # choose by depth
            un_depth_img = depth_list[np.argmin(ave_depth_list)]
            
            # choose by center
            # un_depth_img = depth_list[np.argmin(mask_center_list)]

            print("4 time: {}".format(time.time() - start_time))
        else:
            """
            project pointcloud back to plane
            """
            previews_pointcloud = o3d.geometry.PointCloud()
            previews_points = np.asarray(self.target_pointcloud.points).copy()
            previews_pointcloud.points = o3d.utility.Vector3dVector(previews_points)
            previews_pointcloud.transform(T_inv)
            target_points_cam = np.asarray(previews_pointcloud.points)
            fx = self.o3d_camera_intrinsic.intrinsic_matrix[0, 0]
            fy = self.o3d_camera_intrinsic.intrinsic_matrix[1, 1]
            cx = self.o3d_camera_intrinsic.intrinsic_matrix[0, 2]
            cy = self.o3d_camera_intrinsic.intrinsic_matrix[1, 2]
            projected_target_pixel = []
            for i in range(len(target_points_cam)):
                u = target_points_cam[i, 0] * fx / target_points_cam[i, 2] + cx
                v = target_points_cam[i, 1] * fy / target_points_cam[i, 2] + cy
                if 0 <= u < 480 and 0 <= v < 640:
                    projected_target_pixel.append([u, v])
            projected_target_pixel = np.asarray(projected_target_pixel)
            projected_target_pixel = projected_target_pixel.astype(int)
            # projected_target_pixel = np.asarray(projected_target_pixel[:, :2])
            pixel_set, _ = np.unique(projected_target_pixel, axis=0, return_index=True)
            # print("pixel_set: {}".format(pixel_set))
            target_area = np.zeros((480, 640), dtype=int)
            target_area[pixel_set[:, 1], pixel_set[:, 0]] = 1  # u and v start with 1

            # # # print("target_area: {}".format(target_area))
            # cv2.imshow('Image', target_area.astype(np.uint8)*255)
            # # Wait for a key press to exit
            # cv2.waitKey(0)
            # # Close all windows
            # cv2.destroyAllWindows()


            """
            compare current mask and projected target_mask
            """

            overlap_list = []
            for i in range(len(pred_visibles)):
                mask = np.array(pred_visibles[i]).astype(int)
                overlap = np.logical_and(mask, target_area)
                overlap_list.append(np.count_nonzero(overlap)/np.count_nonzero(mask))

            kernel = np.ones((3, 3), np.uint8)
            final_mask = np.array(pred_visibles[np.argmax(overlap_list)]).astype(np.uint8)
            final_mask = cv2.erode(final_mask, kernel, iterations=1)
            # final_mask = np.array(pred_visibles[0]).astype(int)  # randomly choose, remember to delete
            un_depth_img = unnormalize_depth(depth_img)
            un_depth_img[np.logical_not(final_mask)] = 0
            # kernel = np.ones((3, 3), np.uint8)
            # un_depth_img = cv2.erode(un_depth_img, kernel, iterations=1)
            print("4 time: {}".format(time.time() - start_time))

        o3d_rgb_img = o3d.geometry.Image(rgb_img)
        o3d_depth_img = o3d.geometry.Image(un_depth_img)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb_img, o3d_depth_img)
        o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(
                                            rgbd_image, self.o3d_camera_intrinsic)

        sampled_pc, _ = o3d_pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        if self.target_pointcloud is None:
            sampled_pc.transform(T)

            # visualize part
            if self.vis:
                o3d.visualization.draw_geometries([sampled_pc])
            self.target_pointcloud = o3d.geometry.PointCloud()
            self.target_pointcloud.points = o3d.utility.Vector3dVector(sampled_pc.points)
        else:
            sampled_pc.transform(T)
            # visualize part
            self.target_pointcloud.paint_uniform_color([1, 0, 0])
            sampled_pc.paint_uniform_color([0, 1, 0])
            if self.vis:
                o3d.visualization.draw_geometries([self.target_pointcloud, sampled_pc])

            # sampled_pc = sampled_pc.uniform_down_sample(2+self.count)
            print(len(sampled_pc.points))
            downsample_index = np.random.choice(int(len(sampled_pc.points)), int(int(len(sampled_pc.points)) * (0.95 ** self.count)), replace=True)
            sampled_pc.points = o3d.utility.Vector3dVector(np.asarray(sampled_pc.points)[downsample_index, :])
            print(len(sampled_pc.points))
            merged_pcd = self.target_pointcloud + sampled_pc
            # merged_pcd_downsampled = merged_pcd.voxel_down_sample(voxel_size=0.005)
            # o3d.visualization.draw_geometries([merged_pcd_downsampled])
            # self.target_pointcloud = o3d.geometry.PointCloud()
            # self.target_pointcloud.points = o3d.utility.Vector3dVector(merged_pcd_downsampled.points)

            # set the merged_pcd's points to a fixed number, 40000 is recommanded
            fixed_size_index = np.random.choice(int(len(merged_pcd.points)), 4096, replace=True)
            merged_pcd.points = o3d.utility.Vector3dVector(np.asarray(merged_pcd.points)[fixed_size_index, :])

            if self.vis:
                o3d.visualization.draw_geometries([merged_pcd])
            self.target_pointcloud = o3d.geometry.PointCloud()
            self.target_pointcloud.points = o3d.utility.Vector3dVector(merged_pcd.points)

        # self.target_pointcloud = o3d.geometry.PointCloud()
        # self.target_pointcloud.points = o3d.utility.Vector3dVector(sampled_pc.points)

        # pc = PointCloud2()
        header = Header()
        header.stamp = rospy.Time().now()
        header.frame_id = "base"
        pc = create_cloud_xyz32(header, np.asarray(self.target_pointcloud.points))
        # pc.fields = [
        #     PointField('x', 0, PointField.FLOAT32, 1),
        #     PointField('y', 4, PointField.FLOAT32, 1),
        #     PointField('z', 8, PointField.FLOAT32, 1)]
        # pc.height = 1
        # pc.width = len(np.asarray(self.target_pointcloud.points))
        # pc.is_bigendian = False
        # pc.point_step = 12
        # pc.row_step = pc.point_step * len(np.asarray(self.target_pointcloud.points))
        # pc.is_dense = False
        # pc.data = np.asarray(self.target_pointcloud.points, np.float32).tostring()
        # for i in range(10):
        self.points_pub.publish(pc)
        #     time.sleep(1)

        """
        end of test
        """

        end_time = time.time()
        print("total time: {}".format(end_time - start_time))

        """
        End of add by Matt
        """
        # return results
        return pc


if __name__ == '__main__':

    uoais = UOAIS()
    rospy.spin()
