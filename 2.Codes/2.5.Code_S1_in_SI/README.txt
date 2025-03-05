# Tail_Plume_Calibration

This python script is used for directly calculating the corresponding 2D plume parameters for approximating a given 3D plume evolution with a continuous tail. The script takes inputs via keyboard for parameters and outputs results through console messages and a generated calibration illustration.

## Prerequisites

Ensure you have Anaconda installed on your system. Anaconda is used for managing environments and dependencies.（https://www.anaconda.com/download/success）
To get started, create an environment and install the required dependencies:

Firstly, Open your terminal. 

* Create a new conda environment (name it as you prefer, e.g., 'MPC_env')
>> conda create --name MPC_env python=3.8

* Activate the environment
>> conda activate MPC_env

* Install required packages
>> conda install numpy matplotlib scipy

## Usage

Navigate to the directory where the script withTail_Plume_Calibration.py is located.
>> cd PATH_to_Code_S1

Execute the script using Python 3 with the following command:
>> python3 ./withTail_Plume_Calibration.py

## Inputs
The script will prompt you to enter the parameters for the 3D mantle plume's diameter (d_3D) and the initial temperature anomaly (ΔT_3D). Please follow the on-screen instructions for input.

## Outputs
* Console Output: The results of the computations are displayed in the terminal.
* Illustration: A calibration image (./Calibration.png) will be generated to illustrate the calibration process.

If you have any issues or need more details, please contact zhangruimin22@mails.ucas.ac.cn. Happy modeling!