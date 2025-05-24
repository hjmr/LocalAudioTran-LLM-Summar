import logging
from typing import Dict
import requests
import json

logger = logging.getLogger(__name__)


class SummarizationService:
    def __init__(self):
        self.ollama_url = "http://ollama:11434"
        self.model_name = "phi3.5"
        self.model = None
        # Load model on initialization
        self.load_model()

    def health_check(self) -> bool:
        """Check if model is loaded and Ollama is responsive"""
        try:
            # Check if Ollama is running and model exists
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"] == self.model_name for model in models)
            return False
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    def load_model(self):
        """Create the Phi model in Ollama"""
        try:
            # First check if model already exists
            if self.health_check():
                logger.info("Model already exists")
                self.model = True  # Just set a flag that model is available
                return

            logger.info("Creating Phi model in Ollama...")
            modelfile = '''
FROM phi3.5:3.8b-mini-instruct-q8_0
PARAMETER temperature 0.7
PARAMETER num_ctx 131072
PARAMETER num_gpu 50
TEMPLATE """
{{- if .System }}
{{.System}}
{{- end }}

{{.Prompt}}
"""
'''
            response = requests.post(
                f"{self.ollama_url}/api/create",
                json={
                    "name": self.model_name,
                    "modelfile": modelfile,
                },
            )
            if response.status_code == 200:
                logger.info("Model created successfully")
                self.model = True  # Set flag that model is available
            else:
                logger.error(f"Failed to create model: {response.text}")
                raise Exception(f"Failed to create model: {response.text}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.exception("Full traceback:")
            raise

    def generate_summary(self, text: str) -> Dict:
        """Generate summary using Ollama"""
        try:
            if not text:
                return {
                    "overview": "No text provided",
                    "main_points": [],
                    "key_insights": [],
                    "action_items_decisions": [],
                    "open_questions_next_steps": [],
                    "conclusions": [],
                    "full_text": "",
                }

            # Updated system prompt with recommended sections and instructions
            system_prompt = """
あなたは、トレーニングセッション、ビジネス会議、技術的ディスカッションなど、さまざまな文脈からの音声文字起こしを要約することに特化したAIアシスタントです。提供されたテキストから、**簡潔で明確、かつ実行可能な議事録**を作成してください。

---

## 手順

1. **議事録の中で議論された重要なトピック、タスク、意思決定、潜在的リスクや影響**を特定してください。
2. **情報をでっち上げたり、推測を含めたりしないでください**。内容が不明瞭な場合は、その旨を明記してください。
3. **以下のような重要な情報は省略せずに保持してください**：
   - 関係者の名前または役職（責任者）
   - 日付・締切・マイルストーン（可能な場合）
   - 明示された金額や技術的数値などのデータ

---

## セクション構成（以下の見出しを**そのまま使用してください**）

### 1. **概要（Overview）**
- 会議の全体的な目的と背景（1～3文程度）

### 2. **主な議題（Main Points）**
- 重要なトピックや内容を箇条書きで列挙

### 3. **重要なポイント（Key Insights）**
- 得られた知見、影響、リスクを箇条書きで記載

### 4. **アクションアイテム／意思決定（Action Items / Decisions）**
- 各タスクについて、可能であれば**責任者（owner）**と**期限（deadline）**を明記
- 会議中に下された**明確な意思決定**を記載

### 5. **未解決の問題／次のステップ（Open Questions / Next Steps）**
- 未解決事項や今後の対応事項を箇条書きで列挙
- 所有者や期限が判明していれば、それも記載

### 6. **結論（Conclusions）**
- 主要な成果や最終的なまとめを簡潔に記載

---

## 書式ルール

- 各セクションは**見出し付きで明確に記載**してください。
- セクション内は**箇条書き**または簡潔な段落でまとめてください。
- **重複した情報は繰り返さない**でください。
- **簡潔かつ要点を押さえた表現**を心がけてください。
- **可能な限り短く、しかし必要な情報は網羅する**ことを優先してください。

---

## 補足ガイドライン

- **費用面・スケジュールの遅延など、潜在的リスクや影響**は明記してください。
- **文字起こしに含まれていない憶測的情報は一切含めないでください**。
- **機微な情報（個人情報、金額など）**は文字起こしに明示的に含まれている場合のみ記載してください。
- **不明な点や欠落している情報があれば、その旨を明記**し、内容を補完しないでください。
"""

            logger.info("Generating summary with Ollama...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "system": system_prompt,
                    "prompt": f"次のテキストを要約してください:\n\n{text}",
                    "stream": False,
                    "options": {"temperature": 0.7, "num_ctx": 131072, "num_gpu": 50},
                },
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.text}")

            summary_text = response.json()["response"]
            logger.info(f"Generated summary text: {summary_text}")

            # Prepare our results structure with the new sections
            sections = {
                "overview": "",
                "main_points": [],
                "key_insights": [],
                "action_items_decisions": [],
                "open_questions_next_steps": [],
                "conclusions": [],
                "full_text": summary_text,
            }

            # Parsing the summary text
            current_section = None
            current_points = []

            for line in summary_text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                lower_line = line.lower()

                if "overview:" in lower_line:
                    # Store what we have in current_points before moving on
                    if current_section == "main_points":
                        sections["main_points"] = current_points
                    elif current_section == "key_insights":
                        sections["key_insights"] = current_points
                    elif current_section == "action_items_decisions":
                        sections["action_items_decisions"] = current_points
                    elif current_section == "open_questions_next_steps":
                        sections["open_questions_next_steps"] = current_points
                    elif current_section == "conclusions":
                        sections["conclusions"] = current_points

                    current_section = "overview"
                    current_points = []
                    # Capture text after 'Overview:'
                    sections["overview"] = line.split(":", 1)[1].strip()

                elif "main points:" in lower_line:
                    # Store the previous section's content
                    if current_section == "overview":
                        # Already stored overview text
                        pass
                    elif current_section == "key_insights":
                        sections["key_insights"] = current_points
                    elif current_section == "action_items_decisions":
                        sections["action_items_decisions"] = current_points
                    elif current_section == "open_questions_next_steps":
                        sections["open_questions_next_steps"] = current_points
                    elif current_section == "conclusions":
                        sections["conclusions"] = current_points

                    current_section = "main_points"
                    current_points = []

                elif "key insights:" in lower_line:
                    # Store current main_points
                    if current_section == "main_points":
                        sections["main_points"] = current_points
                    elif current_section == "overview":
                        pass
                    elif current_section == "action_items_decisions":
                        sections["action_items_decisions"] = current_points
                    elif current_section == "open_questions_next_steps":
                        sections["open_questions_next_steps"] = current_points
                    elif current_section == "conclusions":
                        sections["conclusions"] = current_points

                    current_section = "key_insights"
                    current_points = []

                elif "action items / decisions:" in lower_line:
                    # Store key_insights
                    if current_section == "key_insights":
                        sections["key_insights"] = current_points
                    elif current_section == "main_points":
                        sections["main_points"] = current_points
                    elif current_section == "overview":
                        pass
                    elif current_section == "open_questions_next_steps":
                        sections["open_questions_next_steps"] = current_points
                    elif current_section == "conclusions":
                        sections["conclusions"] = current_points

                    current_section = "action_items_decisions"
                    current_points = []

                elif "open questions / next steps:" in lower_line:
                    # Store action_items_decisions
                    if current_section == "action_items_decisions":
                        sections["action_items_decisions"] = current_points
                    elif current_section == "key_insights":
                        sections["key_insights"] = current_points
                    elif current_section == "main_points":
                        sections["main_points"] = current_points
                    elif current_section == "overview":
                        pass
                    elif current_section == "conclusions":
                        sections["conclusions"] = current_points

                    current_section = "open_questions_next_steps"
                    current_points = []

                elif "conclusions:" in lower_line:
                    # Store open_questions_next_steps
                    if current_section == "open_questions_next_steps":
                        sections["open_questions_next_steps"] = current_points
                    elif current_section == "action_items_decisions":
                        sections["action_items_decisions"] = current_points
                    elif current_section == "key_insights":
                        sections["key_insights"] = current_points
                    elif current_section == "main_points":
                        sections["main_points"] = current_points
                    elif current_section == "overview":
                        pass

                    current_section = "conclusions"
                    current_points = []

                elif line.startswith("-") and current_section in [
                    "main_points",
                    "key_insights",
                    "action_items_decisions",
                    "open_questions_next_steps",
                    "conclusions",
                ]:
                    current_points.append(line.lstrip("- ").strip())
                else:
                    # If it's an overview or a line that doesn't match the above sections
                    if current_section == "overview" and sections["overview"]:
                        # Append to the overview text
                        sections["overview"] += " " + line
                    elif current_section in [
                        "main_points",
                        "key_insights",
                        "action_items_decisions",
                        "open_questions_next_steps",
                        "conclusions",
                    ]:
                        # If we want to treat these as bullet items
                        current_points.append(line.strip())

            # After the loop, store the last batch of points if needed
            if current_section == "main_points":
                sections["main_points"] = current_points
            elif current_section == "key_insights":
                sections["key_insights"] = current_points
            elif current_section == "action_items_decisions":
                sections["action_items_decisions"] = current_points
            elif current_section == "open_questions_next_steps":
                sections["open_questions_next_steps"] = current_points
            elif current_section == "conclusions":
                sections["conclusions"] = current_points

            return sections

        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            logger.exception("Full traceback:")
            return {
                "overview": f"Error in summary generation: {str(e)}",
                "main_points": [],
                "key_insights": [],
                "action_items_decisions": [],
                "open_questions_next_steps": [],
                "conclusions": [],
                "full_text": f"Error in summary generation: {str(e)}",
            }
