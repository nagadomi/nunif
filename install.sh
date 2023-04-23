echo "Install dependencies packages"
echo "Install List: [git-core, libmagickwand-dev, libsnappy-dev, libraqm-dev, python-is-python3, python3, python3-pip, python3-dev, python3-tk]"
echo "Do you want to continue?"
echo -n "Yes or No: "
read agreement
if [ $agreement = "yes" ]
then
    sudo apt update -y
    sudo apt upgrade -y
    sudo apt install git-core libmagickwand-dev libsnappy-dev libraqm-dev python-is-python3 python3 python3-pip python3-dev python3-tk -y
    echo "Done!"
elif [ $agreement = "y" ]
then
    sudo apt update -y
    sudo apt upgrade -y
    sudo apt install git-core libmagickwand-dev libsnappy-dev libraqm-dev python-is-python3 python3 python3-pip python3-dev python3-tk -y
    echo "Done!"
else
    echo "Exit!"
fi
