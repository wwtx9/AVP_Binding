class MapPoint:
    def __init__(self, mId, FirstSeenFrame, worldPosition, worldPostion_gt):
        self.mId = mId
        self.FirstSeenFrame = FirstSeenFrame
        self.worldPosition = worldPosition
        self.worldPostion_gt = worldPostion_gt
        self.observations = {}

    def setWorldPos(self, worldPos):
        self.worldPosition = worldPos

    def getWorldPos(self):
        return self.worldPosition

    def addObservation(self, CurrentFrame, index):
        if self.observations.__contains__(CurrentFrame) is False:
            self.observations[CurrentFrame] = index


