# Project-001
### Here are some links to YouTube videos and documents that will be of interest
#
### License Plate Detection & Recognition with YOLOv10 and PaddleOCR | Save Data to SQL Database
### https://www.youtube.com/watch?v=Pdem2LmaVJA&list=PLjnfYL2ZgjqwImWxSk6fw457V2mwEjGu9
#
#
### Comprehensive review on vehicle Detection, classification, and counting on highways
### https://www.sciencedirect.com/science/article/pii/S0925231223007506?via%3Dihub
#
#
### data files: the link below is to the data files needed to run the code
#### https://1drv.ms/f/c/09755aa31ee1ce30/EjDO4R6jWnUggAm2YwAAAAAB3_kYV8s3W2wcnlyMlM7JwA
#### Cashmere.MP4 - this is the primary video file we will use for the code
#### Additional video files are located in the subdirectory, TrafficVideos
#
#
### Milestones
#
#### - Create vehicle counter(s) (issue #13)
#### - Create traffic signal in upper right of screen (issue 14)
#### - Create database to store metrics and attributes
#### - Create Power Point presentation for demo to Education Foundation Board (presentation in July 2025)
#

# How to run Project001 using Windows with an NVIDIA GPU

To run Project001 on a Windows machine in such a way that it runs as close as possible to the target environment (linux) we will use the Windows Subsystem for Linux (WSL).  This feature of Windows allows native linux to run on top of the Windows kernel and with the right libraries installed these apps can also take advantage of GPU acceleration.  These instructions describe how to set up your WSL environment, install the additional GPU support libraries, set up X window compatibility layer (so linux can display Windows-like windows on the desktop), and finally to install and run Project001.

Doc Note: For prompt examples below, those prefixed with '>' are commands that run with the Windows Command Prompt and those prefixed with '$' are commands that run within the Linux shell.

## Prerequisites

* Windows 10 or greater - Make sure all updates are installed (instructions [here](https://support.microsoft.com/en-us/windows/install-windows-updates-3c5ae7fc-9fb6-9af1-1984-b5e0412c556a))
* NVIDIA GPU - Make sure the latest drivers are installed (can be found [here](https://www.nvidia.com/en-us/drivers/))

(note: You can run without a GPU but the performance will be nearly unusable.  Only the most minimal GPU is required (for these instructions a GTX 980 was used which, as of this writing, is 11 years old!)

### Step 1: Set up WSL

WSL is a feature of Windows that allows native linux code to run on a Windows machine with hardware support

Click the Start Menu, type “features”, it should show “Turn Windows features on or off” in the list, click this.
Make sure the following features are checked:
* Virtual Machine Platform
* Windows Subsytem for Linux

Click OK.  After some time for installation you may need to reboot.

### Step 2: Install Ubuntu on WSL

Click on the Start Menu, type “command prompt”, it should show the “Command Prompt” in the list, right click on this and select “Run as Administrator”

We will want to use WSL2 so set the default with the command:
```
> wsl -–set-default-version 2
```

In the new command prompt window type
```
> wsl -–list -–online
```

(if you get an error about missing the wsl command, go back to Set up WSL)
In this list you should see all the available linux distros.  For our purposes we will use Ubuntu 24.04.  Install with the following command:
```
> wsl -–install -d Ubuntu-24.04
```

During install you will be prompted to create a username and password.  Enter what you like but make sure to remember the login.  After install launch Linux with:

```
> wsl -d Ubuntu-24.04
```

Once installed you will be sitting on a new command prompt which is running Linux.
At the new Linux prompt update the package manager with the following commands:
```
$ sudo apt update

$ sudo apt full-upgrade
```

You should now have an item in your Windows start menu called “Ubuntu 24.04.1 LTS”.  Clicking on this will launch a command line that is running Linux.  __Do this now__.

### Step 3: Install the CUDA development kit

The NVIDIA  CUDA development kit allows the open computer vision libraries we will use later to speak directly to your GPU.

At the Linux prompt that you started from the previous step (Install Ubuntu on WSL) enter the following commands:
```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb

$ sudo dpkg -i cuda-keyring_1.1-1_all.deb

$ sudo apt-get update

$ sudo apt-get -y install cuda-toolkit-12-9
```

This will take some time.
For some reference these commands come from the NVIDIA page [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network). 

### Step 4: Install X11

In order for Linux to create windows that can be displayed along with other native Windows we need to install X11 libraries in Linux.

Install X11 apps with the following command
```
$ sudo apt install x11-apps
```


Now you should have X11 installed, to test it out run the following command:
```
$ xeyes
```
This should create a new window on your Windows desktop with some eyes.

### Step 5: Setup Python

We should now have the proper requirements for running Project001 with GPU acceleration.  To run Project001 we will use a tool called `conda` to:
* set up a python “virtual environment” and 
* install the correct version of python.

From the Linux prompt run the following commands:
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

$ bash Miniconda3-latest-Linux-x86_64.sh
```
(This will prompt you a few times and you should just answer YES to any yes/no questions)

In order for the install to take effect, close your current Linux prompt window and open it again (when you re-open a new prompt you should see “(base)” and the beginning of your prompt).

Now we will create a new virtual environment for Project001 called “work” with the following command:
```
$ conda create -n work python=3.12.3
```


You now have a virtual environment that is set up to use the correct python version, you just need to activate it with the following command:
```
$ conda activate work
```
To make sure everything is set up properly run the following command:
```
$ python -–version
```

You should see 3.12.3 if python is set up correctly.

Note: Whenever you want to work on Project001 you will open a Linux prompt and you will first need to run ‘conda activate work’ as above.

### Step 6: Configure GIT (optional)

If you've already configured git with your ssh key then you can skip this step.

From your Linux prompt run the following command:

```
$ ssh-keygen -t ed25519
```

You should be able to hit enter for the defaults and for now we can leave the passphrase blank.  This will generate a public and private key file.  Let's get the public key with the following commands:

```
$ cd ~/.ssh

$ cat id_ed25519.pub
```

The above will display a line of text that starts with `ssh-ed25519` and ends with your user name.  Use your mouse to select this entire line of text then hit enter to copy it to the clipboard.

Go to your Github keys page [here](https://github.com/settings/keys).

Note: You may need to login.

Click "New SSH key"

For Title you can enter anything but typically you would use the name of your computer and perhaps a reference to wsl, eg: "mycomputer-wsl"

Leave Key type as "Authentication Key".

For the Key, right click on the field and select paste.

Now click Add SSH key and you are ready to start using git from wsl.

(For more detailed instructions and how to be even more secure on setting your SSH key with github you can visit [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent))

### Step 7: Install and Run Project001
Now let's get the code for Project001.  In your Linux prompt go to home directory and run the following:
```
$ git clone git@github.com:jeacpa/Project-001.git 
```

(If you encounter a security error go back to Step 6)

If you haven't done so aleady make sure you have activated your environment with the command:

```
$ conda activate work
```

Install all the dependencies for Project001 with the following commands:
```
$ cd Project-001

$ pip install -r requirements.txt
```

Almost there! The last thing we need to do is download a sample video for Project001 to use.  Use the link above to down the Cashmere.mp4 video.  Once downloaded it should be in your Windows Downloads folder.

We need to copy the video to the Project001 folder.  Open your Downloads folder by clicking the Start Menu and type ‘\’ then hit enter.  Click on Downloads in the left list, right click on Cashmere.mp4 and select Copy.

Now we need to open the Project001 folder that exists in your wsl environment, click on the Start Menu, type the following:
```
\\wsl$
```

Hit enter, you should see a new explorer window that contains a folder called ‘Ubuntu-24.04’, double click on this, double click on the ‘home’ folder, double click on the folder with your name, and finally double click on the Project-001 folder.  Right click in this folder and select Paste.

Now all we have to do is run Project001 with the following command:
```
$ python CarTrackingLoop.py
```

You should see a couple of one-time downloads then a new window will appear on your Windows desktop which will be displaying the object recognition example and it should also be using your GPU for best performance!
