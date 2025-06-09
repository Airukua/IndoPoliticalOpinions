import re
from itertools import combinations

class TopicFragmentation:
    def __init__(self, topics_str):
        self.topics_str = topics_str
        self.topic_words = self._extract_topics_words()

    def _extract_words(self, topic_str):
        return set(re.findall(r'"\s*([^"]+)\s*"', topic_str))

    def _extract_topics_words(self):
        topics = re.findall(r'Topic #[0-9]+: ([^"]+(?:(?:"[^"]+")?[^"]*)*)', self.topics_str)
        return [self._extract_words(t) for t in topics]

    def _jaccard_distance(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 0
        return 1 - intersection / union

    def calculate_fragmentation(self):
        distances = []
        for t1, t2 in combinations(self.topic_words, 2):
            distances.append(self._jaccard_distance(t1, t2))
        
        if len(distances) == 0:
            return 0
        return sum(distances) / len(distances)
