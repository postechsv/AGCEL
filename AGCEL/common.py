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

    def __str__(self):
        return f"act('{self.label})" # to be dumped

    def __repr__(self):
        #return f'<label: {self.label}, asubs: {self.asubs}>'
        return f"act('{self.label})"