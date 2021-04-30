class Frame:
    def __init__(self, mId, N, preTrackedKpIndexs,originTrackedKpIndexs, Kps, KpRights, mObsMatrices_array, mPostMatrices_array):
        self.mId = mId
        self.N = N
        self.preTrackedKpIndexs = preTrackedKpIndexs
        self.originTrackedKpIndexs = originTrackedKpIndexs
        self.Kps = Kps
        self.KpRights = KpRights
        self.mObsMatrices_array = mObsMatrices_array
        self.mPostMatrices_array = mPostMatrices_array

    def setPose(self, Rwc, twc):
        self.Rwc = Rwc
        self.twc = twc

    def getRotation(self):
        return self.Rwc
    
    def getTranslation(self):
        return self.twc
