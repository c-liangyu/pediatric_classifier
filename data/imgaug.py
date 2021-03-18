import cv2
import torchvision.transforms as tfs


def Common(image):

    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    img_aug = tfs.Compose([
        tfs.ToTensor()
    ])
    image = img_aug(image)
    return image


def Aug(image):
    img_aug = tfs.Compose([
        tfs.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05),
                         scale=(0.95, 1.05), fillcolor=128)
    ])
    image = img_aug(image)

    return image

def Aug2(image):
    img_aug = tfs.Compose([
        tfs.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05),
                         scale=(0.95, 1.05), fillcolor=128),
        tfs.ToTensor()
    ])
    image = img_aug(image)

    return image

def GetTransforms(image, target=None, type='common'):
    # taget is not support now
    if target is not None:
        raise Exception(
            'Target is not support now ! ')
    # get type
    if type.strip() == 'Common':
        image = Common(image)
        return image
    elif type.strip() == 'None':
        return image
    elif type.strip() == 'Aug':
        image = Aug(image)
        return image
    else:
        raise Exception(
            'Unknown transforms_type : '.format(type))

def GetTransforms2(image, target=None, type='common'):
    # taget is not support now
    if target is not None:
        raise Exception(
            'Target is not support now ! ')
    # get type
    if type.strip() == 'Common':
        image = Common(image)
        return image
    elif type.strip() == 'None':
        return image
    elif type.strip() == 'Aug':
        image = Aug2(image)
        return image
    else:
        raise Exception(
            'Unknown transforms_type : '.format(type))
