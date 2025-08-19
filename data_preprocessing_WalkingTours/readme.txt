
gdpr_blur_faces.sh
https://github.com/shashankvkt/DoRA_ICLR24/tree/main/scripts

We may need to update decode

git clone https://github.com/dmlc/decord.git
cd decord
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ../python
python3 setup.py install --user

