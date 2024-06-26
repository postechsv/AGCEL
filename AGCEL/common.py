class Action():
    def __init__(self, label, asubs):
        self.label = label
        self.asubs = asubs

    def __eq__(self, other):
        if isinstance(other, Action):
            return (self.label == other.label) and (self.asubs == other.asubs)
        return False

    def __hash__(self):
        return hash(self.label)

    def get_str_asubs(self):
        entries = []
        for var, val in self.asubs.items():
            entries.append(f"('{var.getVarName()} <- {val.prettyPrint(0)})")
        return ' ; '.join(entries)

    def __str__(self):
        # to be dumped
        return f"aact('{self.label}, {self.get_str_asubs()})" # concrete
        #return f"aact({self.label}, {self.asubs.prettyPrint(0)})" # abstract

    def __repr__(self):
        #return f'<label: {self.label}, asubs: {self.asubs}>'
        return f"aact('{self.label}, {self.asubs.prettyPrint(0)})"

