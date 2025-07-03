import random

class RandomCycler:
    """
    Maintains an internal duplicate of the provided sequence and returns its elements in a balanced random order.
    For a sequence of n elements and m total retrievals, it guarantees:
     - Each element is returned between m // n and ((m - 1) // n) + 1 times.
     - There will be no more than 2 * (n - 1) other elements between two occurrences of the same element.
    """
    
    def __init__(self, source):
        if len(source) == 0:
            raise Exception("Cannot initialize RandomCycler with an empty collection")
        self.all_items = list(source)
        self.next_items = []
    
    def sample(self, count: int):
        shuffle = lambda l: random.sample(l, len(l))
        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            if not self.next_items:
                self.next_items = shuffle(list(self.all_items))
        return out
    
    def __next__(self):
        return self.sample(1)[0]
