scp -i "$env:USERPROFILE\Downloads\option-data.pem" ubuntu@ec2-47-128-230-60.ap-southeast-1.compute.amazonaws.com:/home/ubuntu/BTCUSDFeaturesdata_5m.csv "$env:USERPROFILE\Downloads"

ssh -i "option-data.pem" source venv/bin/activate python main.py