deactivate
conda deactivate

source /home/greenbaum-gpu/jhahn/python_envs/puzzle_env/bin/activate
source /home/greenbaum-gpu/jhahn/python_envs/brainrender_env/bin/activate
source /home/greenbaum-gpu/jhahn/python_envs/render_env/bin/activate

/home/greenbaum-gpu/jhahn/python_envs/render_env/bin/python
/home/greenbaum-gpu/jhahn/python_envs/puzzle_env/bin/python
cd /home/greenbaum-gpu/jhahn/python_projects/puzzlepp

rm -rdf ~/jhahn/data/shape_dataset/results_render/1

rm -rdf ~/jhahn/data/shape_dataset/output/denoiser


exec -a <Name> python render_results.py

/usr/bin/ffmpeg -framerate 1 -i /home/greenbaum-gpu/jhahn/data/shape_dataset/results_render/1/imgs/%04d.png -pix_fmt yuv420p /home/greenbaum-gpu/jhahn/data/shape_dataset/results_render/1/video.mp4




pip install gpytoolbox tqdm libigl jupyter trimesh libigl lit hydra omegaconf easydict open3d gtsam einops chamferdist
pip install hydra-core --upgrad


denoiser
            lr=2e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,



python train_matching.py --cfg /home/greenbaum-gpu/jhahn/python_projects/Jigsaw/experiments/jigsaw_250e_cosine.yaml




pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118


pip install pytorch-lightning --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning==2.1

pip install torch_cluster



find /usr/ -name cc1plus 2>/dev/null

export PATH="/usr/lib/gcc/x86_64-linux-gnu/12:/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/home/greenbaum-gpu/jhahn/python_envs/blender/lib/linux_x64/dpcpp/lib:/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/greenbaum-gpu/jhahn/python_envs/build_linux_bpy/bin:$PYTHONPATH"

sudo apt-get install g++

sudo update-alternatives --remove-all gcc 
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10

pip install torch_cluster-1.6.3+pt20cu118-cp39-cp39-linux_x86_64.whl



rendering should be run on a separate env with Python 3.10
	bpy

install blender

git clone https://projects.blender.org/blender/blender.git
make bpy
source /opt/intel/oneapi/setvars.sh
ln -s /home/greenbaum-gpu/jhahn/python_envs/blender/lib/linux_x64/dpcpp/lib/libsycl.so.7 ./libsycl.so.7




B = the number of instances, N = max number of parts








/users/j/a/jahn25/puzzlepp
/work/users/j/a/jahn25
/work/users/j/a/jahn25/bio-dataset/data/


/users/j/a/jahn25/ffmpeg-7.0.1/ffmpeg -framerate 15.0 -i /work/users/j/a/jahn25/bio-dataset/results_render/1/imgs/%04d.png  -pix_fmt yuv420p /work/users/j/a/jahn25/bio-dataset/results_render/1/video.mp4


/work/users/j/a/jahn25/breaking-bad-dataset/data/everyday/BeerBottle/2927d6c8438f6e24fe6460d8d9bd16c6/fractured_0/

module add cuda/11.8
module use /users/j/a/jahn25/modulefiles

module add puzzlepp_render/1.0.0

module add puzzlepp/1.0.0
source /users/j/a/jahn25/puzzlepp_render_env/bin/activate
pip install -r /users/j/a/jahn25/puzzlepp/renderer/requirements.txt 


module add python/3.11.9
virtualenv puzzlepp_env


pytorch3d는 sbatch를 통해서 GPU 상에서 설치


Download the breaking data set and structure folders appropriately

python /home/greenbaum-gpu/jhahn/python_projects/Breaking-Bad-Dataset.github.io/decompress.py --data_root /home/greenbaum-gpu/jhahn/data/breaking_bad --subset everyday  

/users/j/a/jahn25/puzzlepp_env/bin/python /work/users/j/a/jahn25/breaking-bad-dataset/decompress.py --data_root /work/users/j/a/jahn25/breaking-bad-dataset/data --subset everyday  --category Ring


jupyter nbconvert generate_pc_data.ipynb --to python
python generate_pc_data.py


du -h --max-depth=1 /work/users/j/a/jahn25/breaking-bad-dataset/data/everyday | sort -hr



/users/j/a/jahn25/ffmpeg-7.0.1/ffmpeg -framerate 15.0 -i /work/users/j/a/jahn25/bio-dataset/results_render/0/imgs/%04d.png -c:v /users/j/a/jahn25/x264-snapshot-20191217-2245-stable/bin/x264 -pix_fmt yuv420p /work/users/j/a/jahn25/bio-dataset/results_render/0/video.mp4




sbatch /users/j/a/jahn25/puzzlepp/train_vqvae_jhahn.sb

vqvae 결과물
/work/users/j/a/jahn25/bio-dataset/output/autoencoder/shape_epoch100_bs64/training

denoiser 결과물
/work/users/j/a/jahn25/bio-dataset/output/denoiser/shape_epoch100_bs64/training

verifier 결과물
/work/users/j/a/jahn25/bio-dataset/output/verifier/shape_epoch100_bs64/training

verifier training 시 verifier_data 필요함
test 시 denoiser에서 matching_data 필요함


sbatch  -p general -N 1 -n 1  --mem=24g --wrap="python generate_pc_data.py +data.save_pc_data_path=/work/users/j/a/jahn25/breaking-bad-dataset/data/pc_data/everyday/"


sacct -j 49918555 --format=User,JobID,MaxRSS,Start,End,Elapsed

du -h --max-depth=1 /work/users/j/a/jahn25/breaking-bad-dataset/data/everyday/Ring | sort -hr


jigsaw를 이용해 matching data 생성
matching_base_model.py의 622번 줄의 save_dir 수정
dataset_config.py의 BREAKING_BAD.DATA_DIR 폴더에 입력 파일

"/users/j/a/jahn25/puzzlepp/Jigsaw_matching/experiments/jigsaw_4x4_128_512_250e_cosine_everyday.yaml"의 다음을 수정
DATASET
SUBSET
DATA_FN
WEIGHT_FILE
STATS


zip render_trained.zip -r results_trained_render/* -x results_trained_render/*/imgs/**


gdown --id 1hGtvWNZbLItVWHTU_GG7rTnGLEDFZmsp



whatis("Set up environment for Puzzlepp on Python 3.8.8")
conflict("python","anaconda","bioconda")
prepend_path("PATH","/users/j/a/jahn25/puzzlepp_env/bin")
setenv("PYTHONPATH","/nas/longleaf/apps/python/3.8.8/lib/python3.8")
prepend_path("LD_LIBRARY_PATH","/nas/longleaf/apps/python/3.8.8/lib")
setenv("THEANO_FLAGS","cuda.root=$CUDA_ROOT,device=gpu,floatX=float32,blas.ldflags=-lblas -ltatlas -llapack -lgfortran")
help([[ Puzzlepp on python/3.8.8 module
        ****************************************************

          This module sets up the following environment
          variables for Python
              PATH
              PYTHONPATH
              LD_LIBRARY_PATH

        ****************************************************

]])













module use /users/j/a/jahn25/puzzlepp_env
module use /users/j/a/jahn25/modulefiles


conda create --prefix /share/lsmsmart/jahn25/puzzle_jhahn  python=3.8 -y

conda activate /share/lsmsmart/jahn25/puzzle_jhahn


conda install -p /share/lsmsmart/jahn25/puzzle_jhahn jupyter trimesh igl hydra










conda install  -p /share/lsmsmart/jahn25/puzzle_jhahn pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -p /share/lsmsmart/jahn25/puzzle_jhahn pytorch3d
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

git clone https://github.com/krrish94/chamferdist.git
cd chamferdist && python setup.py install





cd /share/lsmsmart/jahn25








bsub -Is -n 1 -W 10  -q gpu  -gpu "num=1:mode=shared:mps=no" bash


#BSUB -n 32 -q  -q gpu -R "select[rtx2080 || gtx1080 || p100 || a30]"

#BSUB -R "select[rtx2080 || gtx1080 || p100 || a30]"
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=yes"

#BSUB -o /share/lsmsmart/jahn25/output.%J -J program_T=200K 

#BSUB -n 1 python hello.py
module load conda 
python hello.py


pip install -r "D:\_gdrive\jupyterhub\puzzlefusion-plusplus\requirements.txt"

pip install imp


pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
