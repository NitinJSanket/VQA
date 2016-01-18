function [mFeaturesCollapsed] = stemWords(cWords, mFeatures)
%% Stemming each word in cWordsActual
cWordsStemmed = cell(size(cWords));
for iter = 1:length(cWords)
    cWordsStemmed{iter} = porterStemmer(cWords{iter});
end

%% Findind duplicate indices after stemming
[~,uniqueIndices] = unique(cWordsStemmed);
duplicateIndices = setdiff(1:length(cWordsStemmed), uniqueIndices)';

%% Collapsing Counts and Removing duplicates
mFeaturesCollapsed = mFeatures;
for iter= 1:length(duplicateIndices)
    % Find all common indices to this duplicate entry
    commonIndices = find(strcmp(cWordsStemmed, cWordsStemmed{duplicateIndices(iter)}));
    
    % Sum all entries in the first common index and set all other common
    % indices to 0 so that they don't sum up again
    mFeaturesCollapsed(:,commonIndices(1)) = sum(mFeaturesCollapsed(:,commonIndices),2);
    mFeaturesCollapsed(:,commonIndices(2:end)) = 0;
end

% Delete duplicate indices
mFeaturesCollapsed(:,duplicateIndices) = [];

end
