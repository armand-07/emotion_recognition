import torch.nn.functional as F
import torch


def soft_label_encoding(y, classes, epsilon=0.1):
    y_soft = y * (1 - epsilon) + epsilon / classes
    return y_soft


def cat2one_hot(cat_emot, number_of_classes, format = 'hard_label', epsilon = 0.1):
    """ Convert the categorical emotion to a one-hot encoding. Depending on the format, the encoding can be a hard or soft label."""
    # Get the encoding of the categorical emotion
    one_hot = F.one_hot(torch.tensor(cat_emot), num_classes=number_of_classes)
    if format == 'hard_label':
        return one_hot
    elif format == 'soft_label':
        return soft_label_encoding(cat_emot, number_of_classes, epsilon)
    else:
        assert False, "The given format is not valid."


def onehot2cat(one_hot):
    """ Convert a one-hot encoding to the standard id categorical encoding."""
    return torch.argmax(one_hot, dim=1)


def cart2polar_encoding(cont_emot):
    # Convert to polar coordinates
    radius = torch.sqrt(cont_emot[:, 0]**2 + cont_emot[:, 1]**2)
    angle = torch.atan2(cont_emot[:, 1], cont_emot[:, 0])
    # Combine radius and angle into a single tensor
    cont_emot = torch.stack((radius, angle), dim=1)
    return cont_emot


def polar2cart_encoding(cont_emot):
    # Convert to cartesian coordinates
    x = cont_emot[:, 0] * torch.cos(cont_emot[:, 1])
    y = cont_emot[:, 0] * torch.sin(cont_emot[:, 1])
    # Combine x and y into a single tensor
    cont_emot = torch.stack((x, y), dim=1)
    return cont_emot

