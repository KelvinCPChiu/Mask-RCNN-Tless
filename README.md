# Faster R-CNN and Mask R-CNN in PyTorch 1.0 for Tless

This is the implementation based on Mask-RCNN for Tless dataset.

![image](https://github.com/KelvinCPChiu/Mask-RCNN-Tless/blob/master/datasets/tless/result/test_01_0080_0002500_540.jpg)

![image](https://github.com/KelvinCPChiu/Mask-RCNN-Tless/blob/master/datasets/tless/result/test_11_0160_0002500_540.jpg)

![image](https://github.com/KelvinCPChiu/Mask-RCNN-Tless/blob/master/datasets/tless/result/test_19_0080_0002500_540.jpg)

Modification : 

    ./data/datasets/__init__.py

    ./data/datasets/build.py

    ./data/datasets/evaluation/__init__.py

    ./data/transforms/__init__.py

    ./data/transforms/transforms.py

    ./data/transforms/build.py

    ./config/defaults.py

    ./config/path_catalog.py


Add: 

    ./demo/predictor_nms.py                  -- based on original predictor.py, added NMS over the predictions

    ./Demo_COCO .py                             -- demo python file without Jupyter Notebook. To be renamed.

    ./Mask_R-CNN_CPU.ipynb                   -- Modification based on Mask_R-CNN_Demo, added the weight loading function for trained model.

Run the following command to train the network for tless. 

    python tools/train_net.py --config-file "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml" DATASETS.TRAIN '("tless_train_datasetv2",)' SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.WARMUP_ITERS 500 SOLVER.MAX_ITER 5000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 12000

Note the dataset evaluation function is still implementating. Error will occur after testing prediction.

For demo, please use the jupyter notebook of Mask_R-CNN_CPU.ipynb.

The datasets directory should use the following format,

    -datasets 
    |-coco
    |-tless
        |-background                   : background dataset for small dataset augmentation.
        |-test_primesense               
        |-train_primesense
        |-result                       : saving directory for demo.
