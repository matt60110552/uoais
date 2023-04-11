# install detectron
cd /home/user/uoais_ws/src/uoais
python3 -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html


# install adetAdelaiDet
git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet
python3 setup.py build develop

# download pretrain model
cd /home/user/uoais_ws/src/uoais
mkdir output
cd output
mkdir R50_rgbdconcat_mlc_occatmask_hom_concat
mkdir R50_depth_mlc_occatmask_hom_concat



cd /home/user/uoais_ws/src/uoais
pip install testresources
pip install open3d --ignore-installed PyYAML
python3 setup.py build develop
