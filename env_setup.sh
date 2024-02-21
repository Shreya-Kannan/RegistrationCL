module load python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -e /home/shreya/scratch/voxelmorph
pip install torch torchvision
pip install numpy==1.23.0
pip install traitlets==5.9.0

python /home/shreya/scratch/voxelmorph/scripts/torch/train.py --img-list '/home/shreya/scratch/voxelmorph/images/AbdomenMRCT/AbdomenMRCT_dataset.json'  --epochs 10

#python /home/shreya/scratch/voxelmorph/scripts/torch/register.py --moving /home/shreya/scratch/voxelmorph/images/AbdomenMRCT/imagesTr/AbdomenMRCT_0006_0001.nii.gz --fixed /home/shreya/scratch/voxelmorph/images/AbdomenMRCT/imagesTr/AbdomenMRCT_0006_0000.nii.gz --model /home/shreya/scratch/models/0010.pt --moved moved.nii.gz --warp warp.nii.gz
python /home/shreya/scratch/voxelmorph/scripts/torch/test.py --json '/home/shreya/scratch/voxelmorph/images/AbdomenMRCT/AbdomenMRCT_dataset.json' --model /home/shreya/scratch/voxelmorph/models/20240131023912/0180.pt