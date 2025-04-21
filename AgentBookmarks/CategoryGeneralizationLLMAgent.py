import textwrap
from typing import List
from typing import Type, Iterable

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import BaseModel, Field

from AgentBookmarks.CategoryMultiPortAggregatorAgent import CategoryMultiPortAggregatorOutput
from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.ListCollectionAgent import ListModel
from agent_config import DUMMY_LLM
from agent_logging import logger, rich_console
from util.LLMSupport import LLMModel, LLMAgentConfig
from util.SchemaUtils import generate_template_json


class NormalisierteKategorie(BaseModel):
    hauptkategorie: str = Field(
        ...,
        description="Die übergeordnete Kategorie des Lesezeichens (z.B. 'Technik')"
    )
    unterkategorie: str = Field(
        ...,
        description="Die spezifische Unterkategorie des Lesezeichens (z.B. 'Mobile Geräte')"
    )
#
# class KategorienZuordnung(BaseModel):
#     alt: str = Field(
#         ...,
#         description="Ursprüngliche Kategoriebezeichnung (z.B. 'Technologie/Smartphones')"
#     )
#     neu: str = Field(
#         ...,
#         description="Normalisierte Kategoriebezeichnung im Format 'Hauptkategorie/Unterkategorie' (z.B. 'Technik/Mobile Geräte')"
#     )

class KategorieGeneralisierungAntwort(BaseModel):
    kategorien: List[NormalisierteKategorie] = Field(
        ...,
        description="Liste der normalisierten Kategorien, bestehend aus Haupt- und Unterkategorie"
    )

    @staticmethod
    def empty() -> "KategorieGeneralisierungAntwort":
        return KategorieGeneralisierungAntwort(
            kategorien=[],
            zuordnungen=[],
        )



class CategoryGeneralizationLLMAgent(ConnectedAgent):
    """
    An agent that calls OpenAI's LLMs to summarize and condense news articles.

    Attributes:
        input_schema (Type[BaseIOSchema]): Expected input schema.
        output_schema (Type[BaseIOSchema]): Expected output schema.
    """
    input_schema = ListModel
    output_schema = KategorieGeneralisierungAntwort


    def __init__(self, config: LLMAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes an LLMAgent instance with OpenAI API configuration.

        Args:
            config (LLMAgentConfig, optional): Configuration for the agent. Defaults to LLMAgentConfig().
        """
        super().__init__(config, uuid)

        self.model: LLMModel = LLMModel(config, self.__class__.__name__)
        self.model.delete_log_files()



    def run(self, params: ListModel) -> BaseIOSchema:
        """
        Processes the user input and returns a structured summary.
        If `DUMMY_LLM` is enabled, returns dummy data.

        Args:
            user_input (Optional[BaseIOSchema], optional): The input data. Defaults to None.

        Returns:
            BaseIOSchema: The processed response from the LLM.
        """
        data: Iterable[CategoryMultiPortAggregatorOutput] = params.data
        result_set = set()
        for item in params.data: # type: CategoryMultiPortAggregatorOutput
            # item: CategoryMultiPortAggregatorOutput = item
            if not item.llm.ist_gueltig:
                if not item.bookmark.folder:
                    continue
                folder_path = item.bookmark.folder.strip()
                # Split at the first slash
                if '/' in folder_path:
                    parts = folder_path.split('/', 1)
                    main_category = parts[0].replace("/", " - ").strip()
                    sub_category = parts[1].replace("/", " - ").strip()
                else:
                    main_category = folder_path.replace("/", " - ").strip()
                    sub_category = ""
            else:
                main_category = item.category.hauptkategorie.replace("/", " - ").strip()
                sub_category = item.category.unterkategorie.replace("/", " - ").strip()
            if not main_category:
                continue
            entry = f"{main_category}/{sub_category}" if sub_category else main_category
            result_set.add(entry)
        result_string = "\n".join(result_set)

        # raise Exception("Called openai llm")




        if DUMMY_LLM:
            logger.info(f"LLM in DUMMY MODE for page")
            result_object = KategorieGeneralisierungAntwort(
                kategorien=[
                    NormalisierteKategorie(hauptkategorie="Technik", unterkategorie="Mobile Geräte"),
                    NormalisierteKategorie(hauptkategorie="Wissenschaft", unterkategorie="Astronomie"),
                ],

            )
            return result_object

        else:
            logger.info(f"LLM call for page category generalizer ")
            sysprompt = None
            userprompt =  self._prompt(result_string)
            try:
                result_object, usage = self.model.execute_llm_schema(sysprompt, userprompt, targetType=self.output_schema, title='Step Generalize')
                rich_console.print("+ Normalisierte Kategorien:")
                for i, kategorie in enumerate(result_object.kategorien, 1):
                    rich_console.print(f"  {i}. Hauptkategorie: {kategorie.hauptkategorie} | Unterkategorie: {kategorie.unterkategorie}")
            except Exception as e:
                logger.error(f"{self.__class__.__name__} failed with {e}")
                result_object = KategorieGeneralisierungAntwort(structure=[], mapping={})
            #raise Exception("LLM DONE")
            return result_object

        return KategorieGeneralisierungAntwort.empty()


    def _prompt(self, data):
        schema_dict = LLMModel.openai_schema(self.output_schema)
        llm_schema = generate_template_json(schema_dict)

        prompt = textwrap.dedent(f"""
Du erhältst eine Liste von Lesezeichen-Kategorien eines Browsers im Format:
```
Hauptkategorie/Unterkategorie
```
Die Idee:
Deine Eingabe ist die jetztige unstrukturierte Lesezeichenstruktur des Nutzers.
Aus deinen Ausgaben wird eine Lesezeichenstruktur für den Browser des Benutzers erstellt.
Diese soll zwei Ebenen haben, die mit einer sinnvollen Anzahl Verzeichnisse bestückt ist,
z.B: 20 Hauptverzeichnisse mit je so 5-10 Unterverzeichnissen.
Ziel ist, dass der Nutzter später in deine Struktur die Lesezeichen sinnvoll einsortieren kann.
Dazu muss deine Ausgabe eine sinnvolle Baumstruktur mit sinnvollen Namen sein. Die Struktur
muss natürlich die Struktur des Nutzters abbilden können. 

Deine Aufgabe ist es deshalb, dieEingabekategorien zu bereinigen, zu organisieren und zu konsolidieren, 
so dass sie für den Einsatz des Nutztersd optimiert ist.
Wichtig ist, die Anzahl Kategorien zu verringern indem gemeinsame Kategorien gefunden werden. 
Diese dienen später dazu die Lesezeichen in die neuen Kategorien einzusortieren.
Ideal ist eine Liste von so 5-20 Hauptkategorien und pro Hauptkategorie rund 2-10 Unterkategorien zu erhalten.
Damit kann man gut umgehen.
Der Benutzer sortiert später seine Lesezeichen in deine Struktur ein.
Deshalb eine vernünftige Anzahl von Hauptkategorien und Unterkategorien pro Eintrag. 
Kein Sinn macht es zu viele Einträge oder nur einen Eintrag pro Hauptkategorie. 
Um solche fast leeren Einträge zu vermeiden, müssern die Kategorien gut zusammengefasst werden.
Stelle dir vor wie du eine benutzterfreundlich komplette Lesezeichenstruktur für den Benutzters erstellst. 

Eine wichtige Bedingung existiert: Es gibt Hauptkategorien mit Benutzternamen, also menschliche Eigennamen,
(wie Tina, Bob, ...). Dort haben diese Benutzter ihre Lesezeichen gepseichert. Es ist Top Priotität,
dass diese Eigennamen bleiben und die Untersturktur unterhalb der Eigennamen gebaut wird. 
Z.B. Tina muss alle Ihre Lesezeichenkategorien unter Tina behalten!
Beipiel: Es gibt also 
<Eigenname 1>/<Subkategorie 1>
<Eigenname 1>/<Subkategorie 2>
<Anderes Thema 1>/<Subkategorie 3>
<Anderes Thema 2>/<Subkategorie 4>
Hier dürfen nur die Subkategorie 1 und Subkategorie 2 unterhalb Eigenname 1 vereinigt werden.
Auf der Rootebene können dann alle anderen vereinigt werden.


---

### Aufgabe:

Erzeuge eine **normalisierte und deduplizierte Kategoriestruktur**, die:

- **ähnliche oder überlappende Unterkategorien gruppiert**
- **klare und konsistente Benennungen verwendet**
- **benutzerspezifische Kategorien bewahrt**
- Alle Eigennamen (z.B. Tom, Tina) in der Hauptkategorie müssen behalten werden und in die Ausgabeliste genauso
übernommen werden.
- ein JSON-Objekt ausgibt mit
  einer Liste von **neuen Haupt-/Unterkategorien**
- Eine vernünftige Anzahl Lesezeichen zur Ablage in einem Browser bereitstellt. D.h. generiere nicht zu viele Kategorien!
- Wichtig ist sehr ähnliche Kategorien zusammenzufassen (z.b. 'Spielzeuge', 'Toys' und 'Spielwaren' müssen zusammengelegt werden,
  da es gleiche Begriffe in zwei Sprachen und ähnliche Begriffe in Deutsch sind.)  
- Es müssen alle komischen Namen und Kategorien entfernt werden. Ziel ist eine Liste
an guten Kategorienamen in deutscher Sprache oder Eigennamen.

Wichtig: Es ist sehr wichtig, dass ähnliche Kategorien zusammengefasst werden oder gute gemeinsame
Kategorien generiert werden. Es ist unbedingt zu vermeiden, dass es zu viele ähnliche Kategorien gibt,
da sonst die Lesezeichen Ordner immer nur wenige Ziellesezeichen beinhalten werden.  
Vermeide zudem unklare Kategorien wie "unfiled", "unsorted", ... und ersetze sie durch gute Namen selbst wenn
die Eingaben dies beinhalten. 

---

### REGELN:

#### 1. **Benutzerspezifische Hauptkategorien beibehalten**
- Wenn die Hauptkategorie ein Benutzername ist (z.B. `Lena`, `Tom`, etc.), darf sie nicht verändert oder zusammengelegt werden.
- Unterkategorien innerhalb eines Benutzers dürfen zusammengelegt oder umbenannt werden.
- Unterkategorien verschiedener Nutzer dürfen niemals zusammengeführt oder vermischt werden.
- Unterkategorien eines Nutzters und einer allgemeinen Kategorie dürfen nie vermischt werden.
- Allgemeine Kategorien dürfen nicht zu einem Nutzer verschoben werden.

#### 2. **Allgemeine Kategorien sollen zusammengelegt werden**
- Nicht-benutzerspezifische Wurzeln wie `Reisen`, `Gesundheit`, `Technik`, `News`, `Wissenschaft`, etc. 
  sollen zusammengeführt werden. Dazu können diese auch auf ähnliche Begriffe umbenannt werden.
- Verwende sinnvolle, verständliche **deutsche Begriffe** – vermeide zu feine Unterteilungen.

#### 3. **Unterkategorien zusammenfassen und umbenennen**
- Doppelte oder ähnliche Unterkategorien müssen konsolidiert werden.
- Achte auf eine einheitliche Benennungslogik (vermeide z.B. sowohl `X – Y` als auch `Y – X`).
- Bevorzuge kurze, prägnante Namen.
- Sehr spezifische Begriffe sollen in allgemeinere überführt werden (z.B. `Reisetipps`, `Reiseplanung`, `Flugbuchung` → `Reiseorganisation`)

#### 4. Stil "/"
- Weder Hauptkategorie noch Unterkategorie darf ein "/" enthalten, da diese der Trenner der Kategorien ist
- Alle Text der Ausgabe soll auf Deutsch erfolgen - ausser die Kategorien sind im deutschen bekannte Namen, Begriffe, Fachbegriffe oder Slang
  (Smartphones, Tablet, Gadget, Yoga, Streaming, ...) 

---

### AUSGABEFORMAT

Gib ein **einziges JSON-Objekt** zurück mit folgender Struktur:
  {{
  "kategorien": [
    {{
      "hauptkategorie": "...",
      "unterkategorie": "..."
    }},
    ...
  ]
}}

Das bedeutet konkret am Beispiel:

{{
  "kategorien": [
    {{
      "hauptkategorie": "Technik",
      "unterkategorie": "Mobile Geräte"
    }},
    {{
      "hauptkategorie": "Lena",
      "unterkategorie": "Yoga"
    }}
  ]
}}

---

### BEISPIEL INPUT:

Reisen/Flugbuchung  
Reisen/Reiseplanung  
Reisen/Reisetipps  
...

---

### BEISPIEL OUTPUT:
{{
  "kategorien": [
    {{
      "hauptkategorie": "Reisen",
      "unterkategorie": "Reiseorganisation"
    }},
   ...
  ]
}}

---

### Aktuelle Eingabedaten
Hier sind nun die Eingabedaten die verarbeitet werden müssen:
---
{data}
---
Starte jetzt mit der Generierung des Outputs:
        """)

        return prompt.strip()
