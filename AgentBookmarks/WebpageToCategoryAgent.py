import textwrap
from typing import Optional, List, Type

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from openai import BaseModel
from pydantic import Field

from AgentFramework.core.ConnectedAgent import ConnectedAgent
from AtomicTools.webpage_scraper.tool.webpage_scraper import WebpageMetadata
from agent_config import DUMMY_LLM
from agent_logging import logger, rich_console
from util.LLMSupport import LLMModel, LLMAgentConfig
from util.SchemaUtils import generate_template_json


# -----------------------------------------------------------------------------
class WebpageToCategoryInput(BaseIOSchema):
    """Schema representing the input from the user to the AI agent."""

    url: str = Field(..., description="The url of the page")
    content: str = Field(..., description="The raw page content.")
    metadata: WebpageMetadata = Field(..., description="Metadata about the scraped webpage.")
    webpage_error: Optional[str] = Field(None, description="The read status of the webpage.")


class BookmarkOutput(BaseModel):
    """Schema representing the structured output of an AI-processed webpage bookmark."""

    titel: str = Field(..., description="Der Titel der gespeicherten Webseite")
    kategorie: str = Field(..., description="Die Hauptkategorie zur Organisation des Lesezeichens (z.B. Programmierung, Gesundheit, Design)")
    unterkategorie: str = Field(..., description="Eine spezifischere Unterkategorie unter der Hauptkategorie (z.B. JavaScript, UX Design)")
    schlagwoerter: List[str] = Field(..., description="Eine Liste von 3–6 Schlagwörtern zur Inhaltszusammenfassung und besseren Auffindbarkeit")
    beschreibung: str = Field(..., description="Eine kurze Zusammenfassung des Seiteninhalts (1–2 Sätze)")
    ist_gueltig: bool = Field(..., description="True, wenn es sich um eine echte Inhaltsseite handelt. False bei Fehlerseiten, leeren Seiten oder Platzhaltern.")


    @staticmethod
    def empty() -> "BookmarkOutput":
        return BookmarkOutput(
            titel="",
            kategorie="",
            unterkategorie="",
            schlagwoerter=[],
            beschreibung="",
            ist_gueltig=False
        )

class WebpageToCategoryAgent(ConnectedAgent):
    """
    An agent that calls OpenAI's LLMs to summarize and condense news articles.

    Attributes:
        input_schema (Type[BaseIOSchema]): Expected input schema.
        output_schema (Type[BaseIOSchema]): Expected output schema.
    """
    input_schema = WebpageToCategoryInput
    output_schema = BookmarkOutput

    def __init__(self, config: LLMAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes an LLMAgent instance with OpenAI API configuration.

        Args:
            config (LLMAgentConfig, optional): Configuration for the agent. Defaults to LLMAgentConfig().
        """
        super().__init__(config, uuid)

        self.model: LLMModel = LLMModel(config, self.__class__.__name__)
        self.model.delete_log_files()




        #print("Prmot",self.sysprompt)



    def run(self, params: WebpageToCategoryInput) -> BaseIOSchema:
        """
        Processes the user input and returns a structured summary.
        If `DUMMY_LLM` is enabled, returns dummy data.

        Args:
            user_input (Optional[BaseIOSchema], optional): The input data. Defaults to None.

        Returns:
            BaseIOSchema: The processed response from the LLM.
        """

        error_status = params.webpage_error

        if DUMMY_LLM:
            logger.info(f"LLM in DUMMY MODE for page {params.metadata.domain}")
            result_object = BookmarkOutput(titel=params.metadata.title,
                                           kategorie="Test42",
                                           unterkategorie="keine",
                                           schlagwoerter=[],
                                           beschreibung="Dummy bookmark categorizaion",
                                           ist_gueltig=True)

            #raise Exception("LL;  TEST")
            return result_object

        if error_status:
            logger.info(f"LLM omitting page {params.metadata.domain} due to error reading it {error_status}")
            result_object = BookmarkOutput.empty()
            result_object.ist_gueltig = False
            return result_object
        else:
            logger.info(f"LLM call for page {params.metadata.domain} ")

            schema_dict = LLMModel.openai_schema(self.output_schema)
            llm_schema = generate_template_json(schema_dict)

            webpage = self.format_bookmark_input_de(params)

            user_prompt =  textwrap.dedent(f"""
            ---
            ### Eingabedaten:
            {webpage}
            ---

            ### Ausgabeformat:
            {llm_schema}
            
            # Regeln:
            Es ist extrem wichtig, dass du sicher bist, dass die Webseite gültige Contentdaten
            enthält. Nur der echte Content der webpage darf analysiert werden. Gibt es irgendeinen
            Hinweis, dass die Eingabeseite eine Cookieseite, Bannerseite, Fehlerseite, techische Hinweisseite
            ist, ist unbedingt ist_gueltig=False zu setzen.
            Ist es unklar unbedingt auch ist_gueltig=False setzen. Es ist für die weitere Verarbeitung
            extrem wichtig, dass keine false postiv klassifiziert werden. Mit false negative können
            wir besser umgehen.
            

            Erzeuge nur deutschen Text.
            Du darfst ausschließlich die JSON-Ausgabe im angegebenen Format erstellen.
            Die JSON Keywords müssen identisch mit denen des Ausgabeformat übereinstimmen.
            Die JSON Typen müssen korrekt sein, insb Listen und Strings müssen genau stimmen. Wenn die Strings
            Anführungszeichen enthalten, müssen diese in der Ausgabe richtig gequoted werden.
            Gib **keinen zusätzlichen Text** vor oder nach dem JSON aus.

            Erstelle die Ausgabe jetzt und beginne mit {{
            """).strip()

            system_prompt_generator = SystemPromptGenerator(
                background=[
                    "Dieser Assistent ist ein intelligenter Lesezeichen-Agent, der Webseiten analysiert, um Lesezeichen sinnvoll zu organisieren.",
                    "Er erhält eine Webseiten-URL mit Metadaten und Inhalt (als Text oder Markdown) und erstellt strukturierte Lesezeichen-Metadaten."
                ],
                steps=[
                    "Analysiere den Inhalt, um das Hauptthema und den Kontext der Seite zu verstehen.",
                    "Bestimme, ob es sich um eine echte Inhaltsseite handelt oder um eine leere Seite, Fehlerseite, Cookie-Hinweisseite, ... (Feld 'ist_gueltig' entsprechend setzen).",
                    "Leite eine passende Hauptkategorie ab (z.B. 'Programmierung', 'Design', 'Gesundheit' usw.) sowie eine spezifischere Unterkategorie.",
                    "Erstelle einen klaren und prägnanten Titel für das Lesezeichen.",
                    "Bestimme 3–6 Schlagwörter, die den Inhalt gut zusammenfassen und bei der Suche oder Filterung helfen.",
                    "Schreibe eine kurze Beschreibung der Seite (1–2 Sätze)."
                ],
                output_instructions=[
                    "Wichtig: Wenn die Webseite nicht gelesen werden kann oder Cookie oder Fehlerdaten enthält muss 'ist_gueltig' auf False gesetzt werden!",
                    "D.h. wenn es irgendein Hinweis gibt, dass der Eingabetext ist keine korrekten Webpage Inhaltsdaten (Cookie, Banner, Leer, ..) muss 'ist_gueltig' auf False gesetzt werden!",
                    "Die Ausgabe muss korrektes JSON gemäß der angegebenen Vorlage sein. Kein zusätzlicher Text vor oder nach dem JSON.",
                    "Alle Felder müssen beschreibend und nützlich für die Organisation einer persönlichen Wissensdatenbank oder Lesezeichenverwaltung sein."
                ]
            )

            sysprompt = system_prompt_generator.generate_prompt()

            try:
                result_object, usage = self.model.hl_pydantic_completions(sysprompt,
                                                                 user_prompt,
                                                                 targetType=self.output_schema,
                                                                 fix_function=self.fix_function,
                                                                 title='Step Bookmark LLM')
                rich_console.print(f"Processing LLM url {params.metadata.domain}, Valid: {result_object.ist_gueltig}")
            except Exception as e:
                logger.error(f"{self.__class__.__name__} failed with E11  {e}")
                result_object = BookmarkOutput(titel=params.metadata.title,
                                               kategorie="",
                                               unterkategorie="",
                                               schlagwoerter=[],
                                               beschreibung="",
                                               ist_gueltig=False)

            return result_object

    @staticmethod
    def fix_function(text: str) -> str:
        text = text.replace('ö', 'oe')
        text = text.replace('ä', 'ae')
        text = text.replace('ü', 'ue')

        return text

    def format_bookmark_input_en(self, params: WebpageToCategoryInput) -> str:
        metadata = params.metadata
        input_str = textwrap.dedent(f"""
        ## Webpage input:
    
        ### Metadata:
        Title: {metadata.title}
        Author: {metadata.author or "N/A"}
        Description: {metadata.description or "N/A"}
        Site Name: {metadata.site_name or "N/A"}
        Domain: {metadata.domain}
    
        ### URL:
        {params.url}
    
        ### Content:
        {params.content}
    """).strip()
        return input_str

    def format_bookmark_input_de(self, params: WebpageToCategoryInput) -> str:
        metadata = params.metadata
        input_str = textwrap.dedent(f"""
        ## Webseiten-Eingabe:

        ### Metadaten:
        Titel: {metadata.title}
        Autor: {metadata.author or "N/A"}
        Beschreibung: {metadata.description or "N/A"}
        Seitenname: {metadata.site_name or "N/A"}
        Domain: {metadata.domain}

        ### URL:
        {params.url}

        ### Inhalt:
        {params.content}
        """).strip()
        return input_str

