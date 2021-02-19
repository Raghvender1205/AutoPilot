## Getting Started
1. Start by downloading the driving dataset from kaggle

https://www.kaggle.com/sameerqayyum/nvidia-self-driving-car-training-set

2. After that in this folder clone the repositries
3. Run <code>loadData.py</code> to make the pickle files
    ```
    python LoadData.py
    ```
4. Run the following File <code>train.py</code>
    ```
    python train.py
    ```
5. After training, run <code>app.py</code> to finally see the output
    ```
    python app.py
    ```

## Requirements
* tensorflow==2.3.1
* opencv-python
* matplotlib
* sklearn

TODO: 
1. Use a simulator (like qira) to test the model
2. Add a better test video
3. Increase the model efficiency
