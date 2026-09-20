"""
Studio Generator - Uses LLM to generate flashcards and quiz questions
"""
from typing import List, Optional
import json
import re

from models.studio_models import (
    Flashcard, FlashcardCreate, FlashcardGenerateRequest,
    Quiz, QuizQuestion, QuizGenerateRequest,
    DifficultyLevel, QuestionType
)
from utils.llm_generator import LLMGenerator
from utils.studio_manager import StudioManager


class StudioGenerator:
    """Generate flashcards and quizzes using LLM"""
    
    def __init__(self, llm_generator: LLMGenerator, studio_manager: StudioManager):
        self.llm = llm_generator
        self.studio = studio_manager
    
    async def generate_flashcards(self, request: FlashcardGenerateRequest) -> List[Flashcard]:
        """Generate flashcards from content using LLM"""
        
        # Gather source content
        content = await self._gather_content(
            request.space_id,
            request.source_type,
            request.source_ids,
            request.text_content
        )
        
        if not content:
            return []
        
        # Create prompt for LLM
        prompt = self._create_flashcard_prompt(content, request.num_cards, request.difficulty)
        
        # Generate flashcards using LLM
        response = await self.llm.generate(prompt, max_tokens=2000)
        
        if not response:
            return []
        
        # Parse flashcards from response
        flashcards = self._parse_flashcards(
            response,
            request.space_id,
            request.source_type,
            request.source_ids,
            request.difficulty
        )
        
        # Save flashcards to storage
        saved_cards = []
        for card_data in flashcards:
            card = self.studio.create_flashcard(card_data)
            saved_cards.append(card)
        
        return saved_cards
    
    async def generate_quiz(self, request: QuizGenerateRequest) -> Optional[Quiz]:
        """Generate a quiz from content using LLM"""
        
        # Gather source content
        content = await self._gather_content(
            request.space_id,
            request.source_type,
            request.source_ids,
            request.text_content
        )
        
        if not content:
            return None
        
        # Create prompt for LLM
        prompt = self._create_quiz_prompt(
            content,
            request.num_questions,
            request.question_types,
            request.difficulty
        )
        
        # Generate quiz using LLM
        response = await self.llm.generate(prompt, max_tokens=3000)
        
        if not response:
            return None
        
        # Parse quiz questions from response
        questions = self._parse_quiz_questions(response, request.question_types, request.difficulty)
        
        if not questions:
            return None
        
        # Create quiz
        from models.studio_models import QuizCreate
        quiz_data = QuizCreate(
            space_id=request.space_id,
            title=request.title,
            description=f"Generated quiz with {len(questions)} questions",
            questions=questions,
            source_type=request.source_type,
            source_ids=request.source_ids
        )
        
        quiz = self.studio.create_quiz(quiz_data)
        return quiz
    
    async def _gather_content(
        self,
        space_id: str,
        source_type: str,
        source_ids: Optional[List[str]],
        text_content: Optional[str]
    ) -> str:
        """Gather content from various sources"""
        
        if text_content:
            return text_content
        
        content_parts = []
        
        if source_type == "notebook" and source_ids:
            # Get notebook entries
            for entry_id in source_ids:
                entry = self.studio.get_notebook_entry(entry_id)
                if entry:
                    content_parts.append(f"# {entry.title}\n\n{entry.content}")
        
        elif source_type == "file" and source_ids:
            # TODO: Integrate with file retriever to get file content
            # For now, just return a placeholder
            content_parts.append("File content retrieval not yet implemented")
        
        return "\n\n---\n\n".join(content_parts)
    
    def _create_flashcard_prompt(self, content: str, num_cards: int, difficulty: DifficultyLevel) -> str:
        """Create prompt for flashcard generation based on user plan"""
        
        prompt = f"""From the following study note, extract {num_cards} flashcards.
Each flashcard must have:
- A concise question (front)
- A precise answer (max 3 sentences) (back)
- A difficulty tag: easy / medium / hard
- A topic tag matching the concept covered

Return ONLY valid JSON, no markdown:
{{
  "flashcards": [
    {{
      "question": "What is the naive assumption in Naive Bayes?",
      "answer": "Every feature is entirely independent of every other feature given the class label.",
      "difficulty": "medium",
      "topic": "Naive Bayes"
    }}
  ]
}}

Note content:
{content[:4000]}
"""
        return prompt

    def _create_quiz_prompt(
        self,
        content: str,
        num_questions: int,
        question_types: List[QuestionType],
        difficulty: DifficultyLevel
    ) -> str:
        """Create prompt for quiz generation"""
        
        types_str = ", ".join(qt.value for qt in question_types)
        
        prompt = f"""Based on the following content, create a quiz with {num_questions} questions.

Content:
{content[:3000]}  # Limit content length

Question types to include: {types_str}
Difficulty level: {difficulty.value}

Format your response as a JSON array of questions, where each question has:
- "question": The question text
- "type": One of: {types_str}
- "options": Array of 4 options (for multiple_choice only)
- "correct_answer": The correct answer
- "explanation": Brief explanation of why this is correct

Example format:
[
  {{
    "question": "What is...",
    "type": "multiple_choice",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "This is correct because..."
  }},
  {{
    "question": "True or False: ...",
    "type": "true_false",
    "options": ["True", "False"],
    "correct_answer": "True",
    "explanation": "This is true because..."
  }}
]

Generate exactly {num_questions} questions:"""
        
        return prompt
    
    def _parse_flashcards(
        self,
        response: str,
        space_id: str,
        source_type: str,
        source_ids: Optional[List[str]],
        difficulty: DifficultyLevel
    ) -> List[FlashcardCreate]:
        """Parse flashcards from LLM response"""
        
        flashcards = []
        
        try:
            # Try to extract JSON from response (look for the "flashcards" array or direct array)
            json_match = re.search(r'\{[\s\S]*"flashcards"[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group(0))
                cards_data = data.get("flashcards", [])
            else:
                # Fallback to pure array
                json_match = re.search(r'\[[\s\S]*\]', response)
                cards_data = json.loads(json_match.group(0)) if json_match else []
                
            for card_data in cards_data:
                if 'question' in card_data and 'answer' in card_data:
                    # map the returned difficulty string to enum if possible
                    diff_val = card_data.get('difficulty', difficulty.value)
                    try:
                        parsed_diff = DifficultyLevel(diff_val.lower())
                    except ValueError:
                        parsed_diff = difficulty
                        
                    flashcards.append(FlashcardCreate(
                        space_id=space_id,
                        question=card_data['question'],
                        answer=card_data['answer'],
                        difficulty=parsed_diff,
                        source_type=source_type,
                        source_id=source_ids[0] if source_ids else None,
                        metadata={"topic": card_data.get('topic', '')} 
                    ))
        except Exception as e:
            print(f"Error parsing flashcards: {e}")
            # Fallback: Try to parse as simple Q&A pairs
            lines = response.split('\n')
            current_question = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Q:') or line.startswith('Question:'):
                    current_question = line.split(':', 1)[1].strip()
                elif line.startswith('A:') or line.startswith('Answer:'):
                    if current_question:
                        flashcards.append(FlashcardCreate(
                            space_id=space_id,
                            question=current_question,
                            answer=answer,
                            difficulty=difficulty,
                            source_type=source_type,
                            source_id=source_ids[0] if source_ids else None
                        ))
                        current_question = None
        
        return flashcards
    
    def _parse_quiz_questions(
        self,
        response: str,
        question_types: List[QuestionType],
        difficulty: DifficultyLevel
    ) -> List[QuizQuestion]:
        """Parse quiz questions from LLM response"""
        
        questions = []
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                questions_data = json.loads(json_match.group(0))
                
                for idx, q_data in enumerate(questions_data):
                    import uuid
                    
                    # Parse question type
                    q_type = QuestionType.MULTIPLE_CHOICE
                    if 'type' in q_data:
                        try:
                            q_type = QuestionType(q_data['type'])
                        except ValueError:
                            q_type = QuestionType.MULTIPLE_CHOICE
                    
                    questions.append(QuizQuestion(
                        id=str(uuid.uuid4()),
                        question=q_data.get('question', ''),
                        type=q_type,
                        options=q_data.get('options'),
                        correct_answer=q_data.get('correct_answer', ''),
                        explanation=q_data.get('explanation'),
                        points=1,
                        difficulty=difficulty
                    ))
        except Exception as e:
            print(f"Error parsing quiz questions: {e}")
        
        return questions
