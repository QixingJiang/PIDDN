from mmcv.runner import BaseModule

from ..builder import IMG_POSTS


@IMG_POSTS.register_module()
class CatchTest(BaseModule):
    def forward(self, inputs):
        result = inputs[0]
        return result