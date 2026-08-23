class Tree:
    def __init__(self, parent=None, solution=None):
        self.parent = parent
        self.solution = solution
        self.children = []

    @classmethod
    def crate(cls):
        return cls(parent=None, solution=None)

    def is_root(self):
        return self.parent is None

    def add_child(self, solution):
        child = Tree(parent=self, solution=solution)
        self.children.append(child)
        return child

    def depth(self):
        d = 0
        node = self
        while not node.is_root():
            d += 1
            node = node.parent
        return d

    def all_nodes(self):
        if not self.is_root():
            yield self
        for c in self.children:
            yield from c.all_nodes()

    def to_flat_list(self, category=None):
        flat = []
        index_of = {}

        def _walk(node):
            if not node.is_root():
                entry = dict(node.solution)
                entry["category"] = category
                entry["depth"] = node.depth()
                entry["parent_index"] = index_of.get(id(node.parent), None)
                index_of[id(node)] = len(flat)
                flat.append(entry)
            for c in node.children:
                _walk(c)

        _walk(self)
        return flat