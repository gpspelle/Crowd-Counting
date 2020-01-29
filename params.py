import sys

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# we can explicitly make assignments on it 
this.input = 'safe/'

this.original_xres = 1632
this.original_yres = 3248

this.new_xres = 1114
this.new_yres = 2217

