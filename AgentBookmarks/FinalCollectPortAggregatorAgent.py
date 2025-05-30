import textwrap
from typing import Dict, Tuple
from typing import List

from atomic_agents.lib.base.base_tool import BaseToolConfig
from pydantic import BaseModel

from AgentBookmarks.CategoryGeneralizationLLMAgent import KategorieGeneralisierungAntwort, NormalisierteKategorie
from AgentBookmarks.FirefoxBookmarkAgent import FirefoxBookmarksOutput, Bookmark
from AgentBookmarks.GenerateCategoryForBookmarkAgent import GenerateCategoryForBookmarkOutput
from AgentBookmarks.config import OFFLINE_BOOKMARK
from AgentFramework.core.ListCollectionAgent import ListModel
from AgentFramework.core.MultiPortAgent import MultiPortAggregatorAgent
from AgentNews.NewsSchema import MergedOutput
from agent_config import DUMMY_LLM
from agent_logging import logger, rich_console
from util.LLMSupport import LLMModel, LLMAgentConfig


class FinalCollectPortAggregatorAgent(MultiPortAggregatorAgent):
    """
    Agent that merges search results, web scraping results, and LLM-generated news into a unified data structure.

    Attributes:
        input_schemas (Dict[str, Type[BaseModel]]): Specifies expected input schemas.
        output_schema (Type[BaseModel]): Defines the expected output schema.
    """
    input_schemas = [
        ListModel,
        KategorieGeneralisierungAntwort,
        FirefoxBookmarksOutput,
    ]
    output_schema = FirefoxBookmarksOutput

    def __init__(self, config: LLMAgentConfig, uuid: str = 'default') -> None:
        """
        Initializes the MultiPortAggregatorAgent.

        Args:
            config (BaseToolConfig): Configuration settings for the agent.

        Raises:
            TypeError: If input_schemas or output_schema are not defined.
        """
        super().__init__(config, uuid=uuid)
        self.model: LLMModel = LLMModel(config, self.__class__.__name__)
        self.model.delete_log_files()

    @staticmethod
    def is_multiline(s):
        non_empty_lines = [line for line in s.splitlines() if line.strip()]
        return len(non_empty_lines) > 1

    def format_category(self, cat_llm: str) -> str:
        """
        Convert an LLM‑supplied category string into
        'Main/Sub‑Part - Deeper - StillDeeper' form.

        Rules:
          • First '/' splits main vs. sub.
          • Any further '/' inside the sub‑part become ' - '.
          • Leading/trailing whitespace is stripped.
          • Each preserved segment is capitalised only on its first letter.
          • Empty, None, or just‑whitespace input → '' (empty string).
        """
        # ---------- early exit ----------
        if not cat_llm or not cat_llm.strip():
            return ""

        def smart_capitalize(text: str) -> str:
            text = text.strip()
            return text and text[0].upper() + text[1:] or ""

        # ---------- split at *first* slash ----------
        main_raw, *sub_raw_list = cat_llm.strip().split("/", 1)
        main_category = smart_capitalize(main_raw)

        # ---------- handle sub‑part (if any) ----------
        if sub_raw_list:
            sub_raw = sub_raw_list[0].strip()
            # split *remaining* slashes, cap each chunk, then join with " - "
            sub_chunks = [smart_capitalize(chunk) for chunk in sub_raw.split("/") if chunk.strip()]
            sub_category = " - ".join(sub_chunks)
        else:
            sub_category = ""

        # ---------- build final path ----------
        return f"{main_category}/{sub_category}" if sub_category else main_category

    def run(self, inputs: Dict[str, BaseModel]) -> BaseModel:
        """
        Merges search results, web scraping data, and LLM-generated content into a single structured output.

        Args:
            inputs (Dict[str, Tuple[str, BaseModel]]): Dictionary containing categorized input data.

        Returns:
            MergedOutput: A structured output containing aggregated information.
        """
        rawCategories: ListModel = inputs[ListModel]
        finalCategories: KategorieGeneralisierungAntwort = inputs[KategorieGeneralisierungAntwort]
        firefoxBookmarks: FirefoxBookmarksOutput = inputs[FirefoxBookmarksOutput]

        listData:List[GenerateCategoryForBookmarkOutput] = rawCategories.data
        categoryList: List[NormalisierteKategorie] = finalCategories.kategorien
        bookmarks:List[Bookmark] = firefoxBookmarks.bookmarks

        categories = []
        for kats in categoryList:
            cat = f"{kats.hauptkategorie}/{kats.unterkategorie}"
            categories.append(cat)

        outputBookmarks = FirefoxBookmarksOutput(bookmarks=[])

        min_len = min(len(listData), len(bookmarks))
        for i in range(min_len):
            bookmark: Bookmark = bookmarks[i]
            newBookmark = bookmarks[i].model_copy(deep=True)  # Deep copy using Pydantic v2

            # Match best category from overall categories
            search = f"{listData[i].hauptkategorie}/{listData[i].unterkategorie}"

            # Check offline pages
            if listData[i].hauptkategorie == OFFLINE_BOOKMARK:
                newBookmark.folder = search
                outputBookmarks.bookmarks.append(newBookmark)
                rich_console.print(f"[green]Found OFFLINE_BOOKMARK for {search} [/green]")
                continue


            old_folder = bookmark.folder

            # Use LLM for match
            if DUMMY_LLM:
                cat_llm = search
            else:
                cat_llm = None
                for tryi in range(1,5):
                    cat_llm = self._find_via_llm(search, old_folder, categories)
                    # Found not match that is ok
                    if not cat_llm:
                        break
                    is_multi = FinalCollectPortAggregatorAgent.is_multiline(cat_llm)
                    # if single line assume it is ok
                    if not is_multi:
                        break
                    rich_console.print(f"[red]LLm did not match category, retrying {tryi} cause we got {cat_llm}[/red]")
                    cat_llm = None
            rich_console.print(f"[green]Found match {i}/{min_len} using llm for {search} in folder {old_folder} renaming to {cat_llm}[/green]")

            # We got result
            if cat_llm:
                newBookmark.folder = self.format_category(cat_llm)
            else:
                logger.warning(f"LLM cannot generate bookmark for {search}")
                for cat in categories:
                    logger.warning(f"  - available would be: {cat}")
                newBookmark.folder = self.format_category(search)
            outputBookmarks.bookmarks.append(newBookmark)
            rich_console.print(f"[green]Append bookmark {i}/{min_len}  {newBookmark.folder} instead of old folder {old_folder} [/green]")

        return outputBookmarks

    def _find_via_llm(self, target:str, folder:str, categories:List[str]) -> str:
        if DUMMY_LLM and False:
                logger.info(f"LLM in DUMMY MODE for boomark search")
                return None
        else:
            logger.info(f"LLM call for bookmark search")
            sysprompt = None
            userprompt =  self.build_llm_match_prompt(target, folder, categories)
            # print("LLM FIND PROMPT",userprompt)
            try:
                result, usage = self.model.create_text_completions(sysprompt, userprompt, temperature=0.1)
                if result:
                    result = result.strip()
                if result and result == "KEINE":
                    result = None
                return result
            except Exception as e:
                logger.error(f"{self.__class__.__name__} failed with {e}")
                return None

    from typing import List

    def build_llm_match_prompt(self, target: str, folder: str, categories: List[str]) -> str:
        categories_block = "\n".join(categories)

        prompt = textwrap.dedent(f"""
Deine Aufgabe besteht darin, eine Benutzereingabe eines neuen Lesezeichens möglichst genau einer bestehenden
Kategorie aus einer vorgegebenen Liste von Browser-Lesezeichen zuzuordnen.
Ziel ist es, dass Lesezeichen in den ähnlichst möglichen Ordner kopiert werden können.

## Eingaben:
### 1. Existierende Kategorien-Liste (aus dieser muss die Ausgabe gewählt werden):
{categories_block}

### 2. Benutzerverzeichnis (als zusätzlicher Kontext um die Intention des Benutzers zu verstehen):
{folder}

### 3. Benutzereingabe (das zu suchende und zu kategorisierende Lesezeichen):
{target}

## Aufgabe:
Ordne die Benutzereingabe der am besten passenden Kategorie aus der vorgegebenen Liste zu. 
Berücksichtige dabei Folgendes:

- Die ausgegebene Kategorie muss **exakt identisch** mit einer Zeile (Kategorie) aus der Liste sein.
- **Erstelle niemals neue Kategorien**, außer in dem einen Ausnahmefall unten beschrieben ('unsortiert').
- Die Eingabe kann ungenau sein, Tippfehler enthalten, in einer anderen Sprache sein oder Synonyme verwenden.
- Die Eingabe kann unter Umständen nur ein ähnliches Thema behandeln. Dann gilt es klug zu matchen.
- Priorisiere die genaue Übereinstimmung von **Hauptkategorien** vor der Übereinstimmung von Unterkategorien.
- Hauptkategorien mit **Eigennamen** (z.B. 'Bob' und als Eingabe 'Bob/Reisen-Hawaii') dürfen **niemals geändert** werden. 
In diesen Fällen immer die exakte Hauptnamenskategorie wählen, und die beste Unterkategorie suchen (z.B. 'Bob/Reisen') 

## Notfall-Sonderregel für komplett fehlende Übereinstimmung:
- Wenn keine passende Unterkategorie existiert, wähle die beste Hauptkategorie 
  und ergänze mit Unterkategorie ''unsortiert' (z.B. „Nachrichten/Unsortiert“).
- Falls absolut keine Hauptkategorie passt, verwende die Kategorie **Unsortiert**.
Wichtig: Diese Notfallregel darf nur angewendet werden, wenn es überhaupt keine ähnlichen oder sinnvoll anpassbaren
Kategorien gibt. Sie dient nur dazu leeren Output zu vermeiden. Zu bevorzugen ist immer ein thematischer ähnlicher Match.

## Vorgehensweise (Priorität):
1. Prüfe zuerst direkte Übereinstimmungen.
2. Prüfe Schreibfehler, ähnliche Namen, Übersetzungen in Fremdsprachen
3. Prüfe Eigennamen der Hauptkategorie
4. Prüfe indirekte oder thematisch ähnliche Übereinstimmungen von Hauptkategorien.
5. Prüfe indirekte oder thematisch ähnliche Übereinstimmungen von Untertkategorien.
6. Prüfe Analogien - wo würde es thematisch einsortiert werden können - berücksichtige Eingabe und Kontext
7. Nutze „unsortiert“ nur, wenn keine bessere Lösung existiert (absoluter Notfall).

Tipp: Berücksichtige das Kontextverzeichnis bei Unsicherheit zur thematischen Einordnung.

## Ausgabeformat:
- Gib **ausschließlich** genau einen Eintrag einer Kategorie der Eingabeliste aus.
- Gib nur eine Zeile, d.h. die Katogorie zurück
- **Keine** zusätzlichen Texte, Kommentare, Quotes oder Formatierungen.

## Beispiel:
**Benutzereingabe:**  
> Sport/Mein Yoga

**Kontext:**  
> Yoga Tutorials

**Kategorien-Liste:**  
- Sport/Yoga-Tutorials
- Reisen/...
- ...

**Ausgabe:**  
Sport/Yoga-Tutorials

Halte dich strikt an diese Anweisungen und gib jetzt die am besten passende Kategorie zurück:
        """).strip()

        return prompt
