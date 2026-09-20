"""
Studio Manager - Handles Notebook, Flashcards, and Quiz storage and operations
"""
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

from models.studio_models import (
    NotebookEntry, NotebookEntryCreate, NotebookEntryUpdate,
    Flashcard, FlashcardCreate, FlashcardUpdate, FlashcardReview,
    Quiz, QuizCreate, QuizResult, QuizHistory, QuizAnswer,
    MasteryLevel, DifficultyLevel
)
import config


class StudioManager:
    """Manages all Studio features: Notebook, Flashcards, Quiz"""
    
    def __init__(self):
        """Initialize studio manager with data directories"""
        self.studio_dir = config.DATA_DIR / "studio"
        self.notebooks_dir = self.studio_dir / "notebooks"
        self.notebook_dir = self.studio_dir / "notebook"
        self.flashcards_dir = self.studio_dir / "flashcards"
        self.quizzes_dir = self.studio_dir / "quizzes"
        self.quiz_results_dir = self.studio_dir / "quiz_results"
        
        # Create directories
        for directory in [self.notebooks_dir, self.notebook_dir, self.flashcards_dir,
                         self.quizzes_dir, self.quiz_results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_notebook_file_path(self, space_id: str) -> Path:
        """Get the metadata file path for a space notebook."""
        return self.notebooks_dir / f"{space_id}.json"

    def ensure_space_notebook(self, space_id: str, space_name: str = "") -> Dict[str, Any]:
        """Create notebook metadata for a space if it does not exist."""
        file_path = self._get_notebook_file_path(space_id)

        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        now = datetime.now().isoformat()
        notebook_name = space_name.strip() if space_name and space_name.strip() else space_id
        notebook_data = {
            "id": space_id,
            "space_id": space_id,
            "name": notebook_name,
            "created_at": now,
            "updated_at": now
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=2)

        return notebook_data

    def get_space_notebook(self, space_id: str) -> Optional[Dict[str, Any]]:
        """Get notebook metadata for a specific space."""
        file_path = self._get_notebook_file_path(space_id)
        if not file_path.exists():
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _derive_title_from_question(self, question: str) -> str:
        """Generate a readable title from a chat question."""
        question = (question or "").strip()
        if not question:
            return "Chat Note"

        # Keep full question text; UI can visually truncate as needed.
        return question.replace('\n', ' ').strip()
    
    # ========================================================================
    # NOTEBOOK OPERATIONS
    # ========================================================================
    
    def create_notebook_entry(self, entry_data: NotebookEntryCreate) -> NotebookEntry:
        """Create a new notebook entry"""
        # Ensure a notebook record exists for this space.
        self.ensure_space_notebook(entry_data.space_id)

        entry = NotebookEntry(
            id=str(uuid.uuid4()),
            **entry_data.dict()
        )
        
        # Save to file
        file_path = self.notebook_dir / f"{entry.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(entry.dict(), f, indent=2, default=str)
        
        return entry

    def create_notebook_entry_from_chat(
        self,
        space_id: str,
        question: str,
        answer: str,
        chat_id: Optional[str] = None,
        assistant_timestamp: Optional[str] = None,
        tags: Optional[List[str]] = None,
        space_name: str = ""
    ) -> NotebookEntry:
        """Create a notebook entry from a chat Q/A pair."""
        self.ensure_space_notebook(space_id, space_name=space_name)

        metadata: Dict[str, Any] = {
            "question": question,
            "assistant_timestamp": assistant_timestamp,
        }
        if chat_id:
            metadata["chat_id"] = chat_id

        entry_data = NotebookEntryCreate(
            space_id=space_id,
            title=self._derive_title_from_question(question),
            content=f"Q: {question.strip()}\n\nA: {answer.strip()}",
            source_type="chat",
            source_id=chat_id,
            tags=tags or ["chat"],
            metadata=metadata
        )

        entry = self.create_notebook_entry(entry_data)

        # Update notebook metadata timestamp.
        notebook_data = self.ensure_space_notebook(space_id, space_name=space_name)
        notebook_data["updated_at"] = datetime.now().isoformat()
        with open(self._get_notebook_file_path(space_id), 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=2)

        return entry
    
    def get_notebook_entry(self, entry_id: str) -> Optional[NotebookEntry]:
        """Get a single notebook entry by ID"""
        file_path = self.notebook_dir / f"{entry_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return NotebookEntry(**data)
    
    def list_notebook_entries(self, space_id: Optional[str] = None) -> List[NotebookEntry]:
        """List all notebook entries, optionally filtered by space"""
        entries = []
        
        for file_path in self.notebook_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                entry = NotebookEntry(**data)
                
                # Filter by space if specified
                if space_id is None or entry.space_id == space_id:
                    entries.append(entry)
            except Exception as e:
                print(f"Error loading notebook entry {file_path}: {e}")
        
        # Sort by updated_at descending
        entries.sort(key=lambda x: x.updated_at, reverse=True)
        return entries
    
    def update_notebook_entry(self, entry_id: str, update_data: NotebookEntryUpdate) -> Optional[NotebookEntry]:
        """Update an existing notebook entry"""
        entry = self.get_notebook_entry(entry_id)
        if not entry:
            return None
        
        # Update fields
        update_dict = update_data.dict(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(entry, key, value)
        
        entry.updated_at = datetime.now()
        
        # Save
        file_path = self.notebook_dir / f"{entry_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(entry.dict(), f, indent=2, default=str)
        
        return entry
    
    def delete_notebook_entry(self, entry_id: str) -> bool:
        """Delete a notebook entry"""
        file_path = self.notebook_dir / f"{entry_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    # ========================================================================
    # FLASHCARD OPERATIONS
    # ========================================================================
    
    def create_flashcard(self, card_data: FlashcardCreate) -> Flashcard:
        """Create a new flashcard"""
        card = Flashcard(
            id=str(uuid.uuid4()),
            **card_data.dict()
        )
        
        # Save to file
        file_path = self.flashcards_dir / f"{card.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(card.dict(), f, indent=2, default=str)
        
        return card
    
    def get_flashcard(self, card_id: str) -> Optional[Flashcard]:
        """Get a single flashcard by ID"""
        file_path = self.flashcards_dir / f"{card_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Flashcard(**data)
    
    def list_flashcards(self, space_id: Optional[str] = None, 
                       mastery: Optional[MasteryLevel] = None) -> List[Flashcard]:
        """List all flashcards, optionally filtered"""
        cards = []
        
        for file_path in self.flashcards_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                card = Flashcard(**data)
                
                # Apply filters
                if space_id and card.space_id != space_id:
                    continue
                if mastery and card.mastery != mastery:
                    continue
                
                cards.append(card)
            except Exception as e:
                print(f"Error loading flashcard {file_path}: {e}")
        
        # Sort by next_review date (cards due for review first)
        cards.sort(key=lambda x: x.next_review or datetime.now())
        return cards
    
    def update_flashcard(self, card_id: str, update_data: FlashcardUpdate) -> Optional[Flashcard]:
        """Update a flashcard"""
        card = self.get_flashcard(card_id)
        if not card:
            return None
        
        # Update fields
        update_dict = update_data.dict(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(card, key, value)
        
        # Save
        file_path = self.flashcards_dir / f"{card_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(card.dict(), f, indent=2, default=str)
        
        return card
    
    def review_flashcard(self, card_id: str, review: FlashcardReview) -> Optional[Flashcard]:
        """Record a flashcard review and update using SM-2 spaced repetition"""
        card = self.get_flashcard(card_id)
        if not card:
            return None
        
        # Update basic review stats
        card.review_count += 1
        if review.quality >= 3:
            card.correct_count += 1
        
        today = datetime.now()
        card.last_reviewed = today
        
        # SM-2 Algorithm implementation
        quality = review.quality
        
        if quality < 3:
            card.interval = 1          # show again tomorrow
            card.repetitions = 0
            
            # Downgrade mastery based on failure
            if card.mastery == MasteryLevel.MASTERED:
                card.mastery = MasteryLevel.REVIEWING
            else:
                card.mastery = MasteryLevel.LEARNING
        else:
            if card.repetitions == 0:
                card.interval = 1
            elif card.repetitions == 1:
                card.interval = 6
            else:
                card.interval = round(card.interval * card.ease_factor)
            card.repetitions += 1
            
            # Upgrade mastery based on string of consecutive successes
            if card.repetitions >= 4:
                card.mastery = MasteryLevel.MASTERED
            elif card.repetitions >= 2:
                card.mastery = MasteryLevel.REVIEWING
            else:
                card.mastery = MasteryLevel.LEARNING

        # Adjust ease factor
        card.ease_factor = max(1.3, card.ease_factor + 0.1 - (5 - quality) * 0.08)
        card.next_review = today + timedelta(days=card.interval)
        
        # Save
        file_path = self.flashcards_dir / f"{card_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(card.dict(), f, indent=2, default=str)
        
        return card
    
    def delete_flashcard(self, card_id: str) -> bool:
        """Delete a flashcard"""
        file_path = self.flashcards_dir / f"{card_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    # ========================================================================
    # QUIZ OPERATIONS
    # ========================================================================
    
    def create_quiz(self, quiz_data: QuizCreate) -> Quiz:
        """Create a new quiz"""
        quiz = Quiz(
            id=str(uuid.uuid4()),
            **quiz_data.dict()
        )
        
        # Save to file
        file_path = self.quizzes_dir / f"{quiz.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(quiz.dict(), f, indent=2, default=str)
        
        return quiz
    
    def get_quiz(self, quiz_id: str) -> Optional[Quiz]:
        """Get a quiz by ID"""
        file_path = self.quizzes_dir / f"{quiz_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Quiz(**data)
    
    def list_quizzes(self, space_id: Optional[str] = None) -> List[Quiz]:
        """List all quizzes, optionally filtered by space"""
        quizzes = []
        
        for file_path in self.quizzes_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                quiz = Quiz(**data)
                
                if space_id is None or quiz.space_id == space_id:
                    quizzes.append(quiz)
            except Exception as e:
                print(f"Error loading quiz {file_path}: {e}")
        
        # Sort by created_at descending
        quizzes.sort(key=lambda x: x.created_at, reverse=True)
        return quizzes
    
    def delete_quiz(self, quiz_id: str) -> bool:
        """Delete a quiz"""
        file_path = self.quizzes_dir / f"{quiz_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def submit_quiz(self, quiz_id: str, answers: List[QuizAnswer]) -> Optional[QuizResult]:
        """Submit quiz answers and calculate results"""
        quiz = self.get_quiz(quiz_id)
        if not quiz:
            return None
        
        # Create answer lookup
        answer_dict = {ans.question_id: ans for ans in answers}
        
        # Calculate results
        total_points = sum(q.points for q in quiz.questions)
        correct_count = 0
        incorrect_count = 0
        earned_points = 0
        detailed_answers = []
        
        for question in quiz.questions:
            user_answer = answer_dict.get(question.id)
            is_correct = False
            
            if user_answer:
                # Normalize answers for comparison
                correct_ans = question.correct_answer.strip().lower()
                user_ans = user_answer.answer.strip().lower()
                
                is_correct = correct_ans == user_ans
                
                if is_correct:
                    correct_count += 1
                    earned_points += question.points
                else:
                    incorrect_count += 1
            else:
                incorrect_count += 1
            
            detailed_answers.append({
                "question_id": question.id,
                "question": question.question,
                "user_answer": user_answer.answer if user_answer else None,
                "correct_answer": question.correct_answer,
                "is_correct": is_correct,
                "explanation": question.explanation,
                "points": question.points if is_correct else 0
            })
        
        # Create result
        result = QuizResult(
            quiz_id=quiz_id,
            submission_id=str(uuid.uuid4()),
            total_questions=len(quiz.questions),
            correct_answers=correct_count,
            incorrect_answers=incorrect_count,
            score_percentage=(correct_count / len(quiz.questions) * 100) if quiz.questions else 0,
            total_points=total_points,
            earned_points=earned_points,
            answers=detailed_answers
        )
        
        # Save result
        result_file = self.quiz_results_dir / f"{result.submission_id}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, indent=2, default=str)
        
        return result
    
    def get_quiz_history(self, quiz_id: str) -> QuizHistory:
        """Get quiz attempt history"""
        quiz = self.get_quiz(quiz_id)
        if not quiz:
            return None
        
        # Load all results for this quiz
        results = []
        for file_path in self.quiz_results_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                result = QuizResult(**data)
                if result.quiz_id == quiz_id:
                    results.append(result)
            except Exception as e:
                print(f"Error loading quiz result {file_path}: {e}")
        
        # Calculate statistics
        scores = [r.score_percentage for r in results] if results else [0]
        
        history = QuizHistory(
            quiz_id=quiz_id,
            space_id=quiz.space_id,
            quiz_title=quiz.title,
            results=results,
            best_score=max(scores),
            average_score=sum(scores) / len(scores) if scores else 0,
            attempts_count=len(results)
        )
        
        return history
