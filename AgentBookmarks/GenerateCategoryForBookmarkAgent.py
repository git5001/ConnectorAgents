from typing import Type

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from openai import BaseModel
from pydantic import Field

from AgentBookmarks.BookmarkMultiPortAggregatorAgent import BookmarkMultiPortAggregatorOutput
from AgentBookmarks.config import OFFLINE_BOOKMARK
from AgentFramework.ConnectedAgent import ConnectedAgent
from agent_config import DUMMY_LLM
from agent_logging import logger
from util.LLMSupport import LLMModel, LLMAgentConfig
from util.SchemaUtils import generate_template_json


# -----------------------------------------------------------------------------


class GenerateCategoryForBookmarkOutput(BaseModel):
    """
    Schema representing the structured output of an AI-processed webpage bookmark,
    with separate fields for main and subcategory.
    """
    hauptkategorie: str = Field(
        ...,
        description="Die übergeordnete Kategorie des Lesezeichens"
    )
    unterkategorie: str = Field(
        ...,
        description="Die spezifische Unterkategorie des Lesezeichens"
    )
    @staticmethod
    def empty() -> "GenerateCategoryForBookmarkOutput":
        return GenerateCategoryForBookmarkOutput(
            hauptkategorie="",
            unterkategorie="",
        )


class GenerateCategoryForBookmarkAgent(ConnectedAgent):
    """
    An agent that combines various inoputs to generate a bookmark vategory

    Attributes:
        input_schema (Type[BaseIOSchema]): Expected input schema.
        output_schema (Type[BaseIOSchema]): Expected output schema.
    """
    input_schema = BookmarkMultiPortAggregatorOutput
    output_schema = GenerateCategoryForBookmarkOutput


    def __init__(self, config: LLMAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes an LLMAgent instance with OpenAI API configuration.

        Args:
            config (LLMAgentConfig, optional): Configuration for the agent. Defaults to LLMAgentConfig().
        """
        super().__init__(config, uuid)

        self.model: LLMModel = LLMModel(config, self.__class__.__name__)
        self.model.delete_log_files()

    def extract_categories_from_folder(self, folder_path: str) -> tuple[str, str]:
        """
        Takes a folder path like 'Main/Sub/Sub2', returns (Main, Sub - Sub2)
        If no slash is present, returns (Main, '')
        """

        def smart_capitalize(text: str) -> str:
            text = text.strip()
            return text and text[0].upper() + text[1:] or ""

        if not folder_path or not folder_path.strip():
            return "", ""

        folder_path = folder_path.strip()

        if '/' in folder_path:
            main_raw, sub_raw = folder_path.split('/', 1)
            main_category = smart_capitalize(main_raw.replace("/", " - "))
            sub_chunks = [smart_capitalize(chunk) for chunk in sub_raw.split("/") if chunk.strip()]
            sub_category = " - ".join(sub_chunks)
        else:
            main_category = smart_capitalize(folder_path.replace("/", " - "))
            sub_category = ""

        return main_category, sub_category

    def run(self, params: BookmarkMultiPortAggregatorOutput) -> GenerateCategoryForBookmarkOutput:
        """
        Processes the user input and returns a structured summary.
        If `DUMMY_LLM` is enabled, returns dummy data.

        Args:
            user_input (Optional[BaseIOSchema], optional): The input data. Defaults to None.

        Returns:
            BaseIOSchema: The processed response from the LLM.
        """

        if DUMMY_LLM:
            logger.info(f"LLM in DUMMY MODE for page {params.webpage.metadata.domain}")
            result_object =  GenerateCategoryForBookmarkOutput(hauptkategorie=f"H:Kinder", unterkategorie=f"U:{params.bookmark.folder}")
            return result_object
        else:
            logger.info(f"LLM call for page {params.webpage.metadata.domain} ")
            sysprompt = None
            userprompt = self._prompt(params)
            try:
                result_object, usage = self.model.execute_llm_schema(sysprompt, userprompt, targetType=self.output_schema, title='Step 1')
                logger.info(f"LLM boomark result folder={params.bookmark.folder} --> '{result_object.hauptkategorie}/{result_object.unterkategorie}'")
            except Exception as e:
                logger.error(f"{self.__class__.__name__} failed for {params.bookmark.folder} with {e}")
                if params.bookmark.folder:
                    main_category, sub_category = self.extract_categories_from_folder(params.bookmark.folder)
                    result_object = GenerateCategoryForBookmarkOutput(hauptkategorie=main_category,unterkategorie=sub_category)
                else:
                    result_object = GenerateCategoryForBookmarkOutput.empty()
            return result_object



        return GenerateCategoryForBookmarkOutput.empty()

    def _create_data_for_llm(self, item):
        output = []

        # Originales Lesezeichen
        output.append("Originaler Lesezeichen Titel: " + str(item.bookmark.title))
        # output.append("Originale URL: " + str(item.bookmark.url))
        output.append("Originaler Lesezeichen Ordner: " + str(item.bookmark.folder))

        if not item.webpage.error:
            # LLM-Zusammenfassung
            output.append("Webseiten Informationen:")
            if item.llm.ist_gueltig:
                output.append("Titel: " + str(item.llm.titel))
                output.append("Hauptkategorie: " + str(item.llm.kategorie))
                output.append("Unterkategorie: " + str(item.llm.unterkategorie))
                output.append("Schlagwörter: " + str(item.llm.schlagwoerter))
                output.append("Beschreibung: " + str(item.llm.beschreibung))
            #output.append("LLM Ist gültig: " + str(item.llm.ist_gueltig))
            else:
                output.append("Webseite Titel: " + str(item.webpage.metadata.title))
                output.append("Webseite Autor: " + str(item.webpage.metadata.author))
                output.append("Webseite Beschreibung: " + str(item.webpage.metadata.description))
                # output.append("Webseite Fehlerstatus: " + str(item.webpage.error if item.webpage.error else "Kein Fehler"))

            # Webseitendaten
            output.append("Domain: " + str(item.webpage.metadata.domain))
            output.append("Webseite Seitenname: " + str(item.webpage.metadata.site_name))
            #output.append("Webseite Titel: " + str(item.webpage.metadata.title))
            #output.append("Webseite Autor: " + str(item.webpage.metadata.author))
            #output.append("Webseite Beschreibung: " + str(item.webpage.metadata.description))
            #output.append("Webseite Fehlerstatus: " + str(item.webpage.error if item.webpage.error else "Kein Fehler"))
        else:
            output.append("Originale URL: " + str(item.bookmark.url))
            output.append("Die Zuordnung muss anhand des Titles, Ordners und der URL erfolgen ")

        output.append(f"Der Ordner ({str(item.bookmark.folder)}) kann immer als Default für die Zuordnung dienen!" )

        return output

    def _prompt(self, item):

        schema_dict = LLMModel.openai_schema(self.output_schema)
        llm_schema = generate_template_json(schema_dict)

        valid_bookmarks = self._create_data_for_llm(item)
        folder =  item.bookmark.folder
        user_prompt = f"""
        Du organisierst Browser-Lesezeichen in eine zweistufige Ordnerstruktur:

        <**Hauptkategorie**> / <**Unterkategorie**>

        Jedes Lesezeichen enthält:
        - Einen ursprünglichen Ordnernamen (dies kann ein Personenname, ein Thema oder unsinnige Informationen sein)
        - Metadaten wie allgemeine Kategorie, Unterkategorie und Tags

        ### Deine Aufgabe für jedes Lesezeichen:
        1. Analysiere die Hauptkategorie und finde heraus ob sie sinnvoll ist, leicht geänderrt werden muss
            oder total neu geschreiben werden muss.
            Hierfür analysierst du alle Informationen aus der Eingabe. 
           a. **Verwende den ursprünglichen Hauptkategorie Ordnernamen als Hauptkategorie**, wenn er sinnvoll ist, 
            sinnvoll ist z.B.:
           - Ein Personenname („Tina“, „Bob“)
           - Ein Thema („Nachrichten“, „Einkaufen“)
           b. **Ersetze die Hauptkategorie** durch eine passende Kategorie, 
            wenn der ursprüngliche Hauptkategoriename bedeutungslos ist oder Schreibfehler enthält,
             z.B. „a“, „abc“, „temp“, „ordner_1“ usw.
           c. Passe den Hauptkategorienamen an, wenn er sinnvoll ist, aber eine ungewöhnliche Schrebiweise hat,
           z.B. "Kinder1" würde "Kinder" werden 
           d. Übersetze Hauptkategorienamen ins deutsche, ausser sie sind Eigennamen
        2. **Erstelle eine prägnante und sinnvolle Unterkategorie**, 
          basierend auf Inhalt, Thema, Zielgruppe oder Format.
          Die Untergruppe kann von dir weitaus mehr geändert werden. Hier ist es wichtig einen
          neuen guten Namen zu bekommen, egal was der original Name war. Der originale Name dient
          allerdings trotzdem als Input.

        ### Regeln
        Alle Kategorien, Hauptkategorie und Unterkategorie müssen Eigennamen oder 
        sinnvolle detusche Wörter sein. Niemals darf das Ergebnis sowas wie "test1",
        "buch2" "classloader" sein.  
        ### Ausgabesprache
        - Alle Haupt- und Unterkategorienanmen müssen deutsch erzeugt werden
          (d.h. nach deutsch übersetzen oder als deutsch generiert werden)

        ### Beispiele:

        **Eingabe:** (Guter Ordnername, da Eigenname)
        Ordnername: Evelyn  
        Titel: Sternenschweif – Wikipedia  
        Hauptkategorie: Kinder- und Jugendliteratur  
        Unterkategorie: Audiobücher und Hörspiele  
        Tags: ["Sternenschweif", "Hörspiel", "Audiobuch", "Pony", "Kinderserie"]

        **Ausgabe:**
        {{
            "hauptkategorie": "Evelyn",
            "unterkategorie": "Hörbücher – Kinder"
        }}

        ---

        **Eingabe:** (unsinniger Ordnername)
        Ordnername: abc  
        Titel: USB-Lampe mit Touchfunktion  
        Hauptkategorie: Technik  
        Unterkategorie: Smart Home  
        Tags: ["Lampe", "LED", "Touch", "USB", "Kinderzimmer"]

        **Ausgabe:**
        {{
            "hauptkategorie": "Technik",
            "unterkategorie": "Intelligente Beleuchtung"
        }}

        ---

        **Eingabe:**
        Ordnername: Nachrichten  
        Titel: Tagesschau aktuell  
        Hauptkategorie: Nachrichten  
        Unterkategorie: Inland  
        Tags: ["Politik", "Deutschland", "Tagesschau", "Nachrichten"]

        **Ausgabe:**
        {{
            "hauptkategorie": "Nachrichten",
            "unterkategorie": "Deutschland"
        }}

        ---

        Verarbeite nun das folgende Lesezeichen:

        ### Eingabe-Lesezeichen:
        {valid_bookmarks}
        
        ### Eingabe Verzeichnis (das ursprüngliche Lesezeichenverzeichnis und hat hohe Priorität):
        {folder}
        
        Dies Lesezeichenverzeichnis ist wichtig zur Analyse ob das Lesezeichen einen Benutzter zugeordnet war.
        Dieses ist auch wichtig falls keine weiteren Informationen über die Webseite vorliegen.
        In dem Fall muss die neue Hauptkategorie dem Benutzter entsprechen und darf nicht geändert werden!

        ### Ausgabeschema:
        {{
            "hauptkategorie": "<Hauptkategorie>",
            "unterkategorie": "<Unterkategorie>"
        }}

        Die Ausgabe muss **reines JSON** sein, exakt nach diesem Schema:
        - Keine erklärenden Texte davor oder danach
        - Kein Markdown, keine Anführungszeichen drumherum
        - Nur ein gültiges JSON-Objekt im gegebenen Ausgabeschema

        Beginne jetzt mit der Generierung:
        """

        return user_prompt


    def debug_categories(self, params: "BookmarkMultiPortAggregatorOutput") -> None:
        """
        Prints out relevant category-related information from each aggregated item
        in the given ListOutputSchema.
        """
        item = params


        # Original bookmark (user-supplied)
        print("Original Bookmark Title:      ", item.bookmark.title)
        print("Original Bookmark URL:        ", item.bookmark.url)
        print("Original Bookmark Folder:     ", item.bookmark.folder)

        # LLM summary (our AI-generated info)
        print("LLM Titel:                    ", item.llm.titel)
        print("LLM Hauptkategorie (kategorie):", item.llm.kategorie)
        print("LLM Unterkategorie:           ", item.llm.unterkategorie)
        print("LLM Schlagwoerter:            ", item.llm.schlagwoerter)
        print("LLM Beschreibung:             ", item.llm.beschreibung)
        print("Webage Ist Gueltig:           ", item.llm.ist_gueltig)

        # Webpage metadata (scraper data)
        print("Webpage Domain:               ", item.webpage.metadata.domain)
        print("Webpage Site Name:            ", item.webpage.metadata.site_name)
        print("Webpage Title:                ", item.webpage.metadata.title)
        print("Webpage Author:               ", item.webpage.metadata.author)
        print("Webpage Description:          ", item.webpage.metadata.description)
        print("Webpage Error status:         ", item.webpage.error if item.webpage.error else "No error")

        print()  # Blank line for readability
