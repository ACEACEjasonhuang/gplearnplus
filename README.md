# gplearnplus
升级后的gplearn， 支持包含时序和截面参数的自定义函数，例如均线

# v1.0
未调试完全， 有bug

# v1.1
处理完funtions模块的问题
调试成功，对于时序自定义函数中的常数参数，需要在函数中做去广播判定

# v1.2
test中加入了自定义函数的定义方法，需要忽略运行时的RuntimeWarning

# v1.3
functions中去掉了对于function.__code__.co_argument的限制
增强对函数修饰器的兼容