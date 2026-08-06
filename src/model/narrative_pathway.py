from typing import Dict

class NarrativePathway:
    id : str
    name: str
    title : str
    description : str

    def __init__(self, id: str, name: str, title : str, description : str):
        self.id = id
        self.name = name
        self.title = title
        self.description = description
    
    def to_dict(self):
        return {
            "id" : self.id,
            "name" : self.name,
            "title" : self.title,
            "description" : self.description
        }

    @staticmethod
    def from_dict(pathway: Dict) -> "NarrativePathway":
        return NarrativePathway(
            id = pathway['id'],
            name = pathway['name'],
            title = pathway['title'],
            description = pathway['description']
        )
    
    def __repr__(self):
        return str(self.to_dict())
