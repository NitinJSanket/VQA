function featuresXNorm = normFeatures(featuresX)
    numCols = size(featuresX,2);
    l2X = repmat(sqrt(sum(featuresX.^2,2)),[1,numCols]);
    featuresXNorm = featuresX./l2X;
end