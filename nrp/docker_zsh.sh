#!/bin/sh

sudo apt-get update
sudo apt-get install -y zsh
wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

# Theme
cd ~/.oh-my-zsh/themes/ && wget https://raw.githubusercontent.com/AmrMKayid/KayidmacOS/master/kayid.zsh-theme

# nrp
cd ~/ && rm ~/.zshrc 
wget -O ~/.zshrc https://raw.githubusercontent.com/AmrMKayid/KayDDRL/master/nrp/.zshrc_nrp 
source ~/.zshrc
