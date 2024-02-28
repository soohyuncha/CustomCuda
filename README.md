# CustomCuda
This is a brief guide on extending your custom cuda kernel to Python/Pytorch module.


## 1. Design and debug logic in .cu file
```bash
cd ./dev
make
./linear
```

## 2. Extend to Python package
```bash
cd ./extension
bash init.sh
pip install .
```

## 3. Verify functionality with Python script
```bash
python test_linear.py
```

## Debug tips
These are some challenges I have faced when trying to extend custom cuda kernel to Python.

* About version: Pytorch <= 2.0.1 and cuda < 12.x. It doesn't support cuda >= 12.x
* Initialize all the memory in cuda kernel properly. Although it won't cause any problem in .cu file debugging, it can generate some trash value when extending to Python script.
* All the input, output tensors should be in cuda device.
* Bound check when accessing array in cuda kernel.
