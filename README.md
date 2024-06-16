# Machine learning proxy server

# Installment
```bash
sudo apt install -y python3
sudo apt install -y python3-pip
sudo apt install -y python3-venv
python3 -m pip install poetry
python3 -m venv venv
source venv/bin/activate
poetry install
```
## Do not forget to put weights in the `weights` folder
weights should be named `best.pth`
```bash
mkdir weights
cp your_weights.pth weights/
```

# Launch
```bash
python backend.py
```
