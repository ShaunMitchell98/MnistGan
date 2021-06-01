def deserialize(objectInput, jsonDictionary):

    for attr, value in objectInput.__dict__.items():
        setattr(objectInput, attr, jsonDictionary[attr])

    return objectInput
