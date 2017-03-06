
### Get project
git clone https://github.com/sagunms/CarND-Vehicle-Detection.git
cd CarND-Vehicle-Detection

### Activate conda environment
source activate carnd-term1

### Download training data
mkdir data
cd data
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
unzip vehicles.zip
unzip non-vehicles.zip
cd ..

### Configure parameters
vim vehicle_lib/config.py

### Train model from downloaded training data
python model.py -m model.mdl

### Run vehicle detection project (output video)
python main.py -m model.mdl -i project_video.mp4 -o annotated_project_video.mp4