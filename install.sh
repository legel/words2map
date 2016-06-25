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
			conda create --name words2map --yes
		fi
	fi
}

# install_developer_libraries_as_needed() {
# 	OS_ARCHITECTURE="$(uname -s)"
# 	if [ $OS_ARCHITECTURE == "Linux" ]; then
# 		echo "$(python -mplatform | grep -qi Ubuntu && sudo apt-get update && sudo apt-get install python-dev libffi-dev libssl-dev libxml2-dev libxslt1-dev || sudo yum update -y && sudo yum install python-devel && sudo yum install libffi-devel)"
# 	fi
# }

install_python_dependencies() {
	if hash conda 2>/dev/null; then
		echo 'Installing Python dependencies for words2map...'
		source activate words2map
		conda install cython scikit-learn gensim seaborn # mongodb --channel spacy --yes spacy
		# install_developer_libraries_as_needed
		pip install hdbscan pattern semidbm # python-levenshtein textacy
		# echo "Downloading English language model from Spacy.io for keyterm extraction in Textacy:"
		# python -m spacy.en.download
	fi	
}

refresh_user_shell() {
	if [ -n "`$SHELL -c 'echo $BASH_VERSION'`" ]; then
  		exec bash
	elif [ -n "`$SHELL -c 'echo $ZSH_VERSION'`" ]; then
  		exec zsh
  	fi
}

install_words2map_if_space_available() {
	MINIMUM_GB_NEEDED=10
	GB_AVAILABLE="$(df -H | grep -vE '^Filesystem' | awk '{ print $4 }' | sed -n 1p | sed 's/G//')"
	if (( GB_AVAILABLE < MINIMUM_GB_NEEDED )); then
		echo "Sorry! $MINIMUM_GB_NEEDED GB of space is needed, but you have only $GB_AVAILABLE GB available. Consider deleting stuff or launching a new computer in the cloud..."
	else
		install_conda_if_needed
		create_conda_environment
		install_python_dependencies
		refresh_user_shell
	fi
}

install_words2map_if_space_available
