from . import models


def build_model(args,cfg,**kwargs):
    model = models.__dict__[args.model](cfg=cfg, num_classes=args.num_classes)
    return model
