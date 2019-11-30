from alpha_pose.core import estimator

param_dict = {
    "input_path": ["https://i.ytimg.com/vi/XGZwjxN3Bns/maxresdefault.jpg", "https://i.ytimg.com/vi/XGZwjxN3Bns/maxresdefault.jpg"],
    "output_path": "./result",
    "fast_inference": 0
}

cuda = 0

param = estimator.HyperParameter(cuda, param_dict)
estimator.estimate(param)
