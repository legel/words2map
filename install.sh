#!/bin/bash

download_miniconda() {
	echo "Downloading Miniconda for Python dependencies..."
	OS_BIT_TYPE="$(uname -m)"
	OS_ARCHITECTURE="$(uname -s)"
	if [ $OS_BIT_TYPE == "i686" ]; then
		OS_BIT_TYPE="x86"
	fi
	if [ $OS_ARCHITECTURE == "Darwin" ]; then
		OS_ARCHITECTURE="MacOSX"
	fi

	MINICONDA_INSTALL_FILE="Miniconda2-latest-$OS_ARCHITECTURE-$OS_BIT_TYPE.sh"
	MINICONDA_DOWNLOAD_URL="https://repo.continuum.io/miniconda/$MINICONDA_INSTALL_FILE"
	$(curl -O $MINICONDA_DOWNLOAD_URL)
	$(chmod +x $MINICONDA_INSTALL_FILE)
}

install_miniconda() {
	echo "Installing Miniconda..."
	echo "$(./$MINICONDA_INSTALL_FILE -b -p $HOME/miniconda)"
	echo "$(rm $MINICONDA_INSTALL_FILE)"	
}

confirm_miniconda_installed() {
    if hash conda 2>/dev/null; then
    	echo "Miniconda installed!"
    else
    	echo "Failed to install Miniconda. Please visit http://conda.pydata.org/docs/install/quick.html to install and then try rerunning this script, making sure that Miniconda is accessible in the PATH"
    fi
}

update_script_startup_file() {
	echo "if [[ \":\$PATH:\" != *\":\$HOME/miniconda/bin:\"* ]]; then" >> $STARTUP_FILE
	echo "  export PATH=\"\$PATH:\$HOME/miniconda/bin\"" >> $STARTUP_FILE
	echo "fi" >> $STARTUP_FILE
}

add_miniconda_to_path() {
	# temporary update to PATH for this script
	export PATH="$PATH:$HOME/miniconda/bin"

	# permanent update to PATH for user's convenience
	if [ -n "`$SHELL -c 'echo $BASH_VERSION'`" ]; then
		STARTUP_FILE="$HOME/.bashrc"
		update_script_startup_file
	elif [ -n "`$SHELL -c 'echo $ZSH_VERSION'`" ]; then
		STARTUP_FILE="$HOME/.zshrc"
		update_script_startup_file
	else
		echo "Couldn't automatically add Miniconda to the PATH of your preferred terminal. We suggest working from Bash or ZShell." 
	fi
}

install_conda_if_needed() {
	if hash conda 2>/dev/null; then
		echo "Miniconda installed!"
	else
		if ping -c 1 google.com >> /dev/null 2>&1; then
			download_miniconda
			install_miniconda
			add_miniconda_to_path
			confirm_miniconda_installed
		else
			echo "Looks like you're offline! Please address this and then try rerunning this script."
		fi
	fi
}

create_conda_environment() {
	if hash conda 2>/dev/null; then
		CONDA_ENVIRONMENTS="$(conda env list)"
		if [[ "$CONDA_ENVIRONMENTS" != *"words2map"* ]]; then
			conda create --name words2map --yes cython scikit-learn gensim seaborn
		fi
	fi
}

install_developer_libraries_as_needed() {
	OS_ARCHITECTURE="$(uname -s)"
	if [ $OS_ARCHITECTURE == "Linux" ]; then
		# currently dumb handling of ubuntu v. other linux distros
		echo "$(sudo apt-get -y update && sudo apt-get -y install python-dev)"
		echo "$(sudo yum update -y && sudo yum install python-devel -y && sudo yum groupinstall "Development Tools" -y)"
	fi
}

install_hdbscan() {
	git clone https://github.com/lmcinnes/hdbscan/archive/master.zip
	cd hdbscan
	
}

install_python_dependencies() {
	if hash conda 2>/dev/null; then
		echo 'Installing Python dependencies for words2map...'
		source activate words2map
		install_developer_libraries_as_needed
		pip install hdbscan pattern semidbm nltk unidecode
	fi	
}

prepare_words2map() {
	echo "$(tar xzf vectors.tar.gz)"
	echo "$(python -m nltk.downloader punkt stopwords)"
	GREEN="\033[0;32m"
	NOCOLOR="\033[0m"
	echo "Everything installed -"
	echo -e "Activate the words2map virtual machine by typing ${GREEN}source activate words2map${NOCOLOR}, then..."
	echo -e "${GREEN}python words2map.py${NOCOLOR} to generate your first map!"
}

refresh_user_shell() {
	if [ -n "`$SHELL -c 'echo $BASH_VERSION'`" ]; then
  		exec bash
	elif [ -n "`$SHELL -c 'echo $ZSH_VERSION'`" ]; then
  		exec zsh
  	fi
}

install_words2map_if_space_available() {
	install_conda_if_needed
	create_conda_environment
	install_python_dependencies
	prepare_words2map
	refresh_user_shell
}

install_words2map_if_space_available
# create and remove swap file if RAM is not large enough for installation, as shown here: http://stackoverflow.com/a/18335151/1241952
