"""
Studio Models - Notebook, Flashcards, Quiz
These models represent the core Studio features for NotebookPRO
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# NOTEBOOK MODELS
# ============================================================================

class NotebookEntry(BaseModel):
    """A single note entry in the notebook"""
    id: str = Field(..., description="Unique identifier for the note")
    space_id: str = Field(..., description="Space this note belongs to")
    title: str = Field(..., description="Title of the note")
    content: str = Field(..., description="Main content/body of the note")
    source_type: str = Field(default="manual", description="Source: manual, chat, generated")
    source_id: Optional[str] = Field(None, description="ID of source (e.g., chat message ID)")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NotebookEntryCreate(BaseModel):
    """Request model for creating a notebook entry"""
    space_id: str
    title: str
    content: str
    source_type: str = "manual"
    source_id: Optional[str] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class NotebookEntryUpdate(BaseModel):
    """Request model for updating a notebook entry"""
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# FLASHCARD MODELS
# ============================================================================

class DifficultyLevel(str, Enum):
    """Difficulty level for flashcards"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class MasteryLevel(str, Enum):
    """User's mastery level for a flashcard"""
    NEW = "new"
    LEARNING = "learning"
    REVIEWING = "reviewing"
    MASTERED = "mastered"


class Flashcard(BaseModel):
    """A single flashcard for memorization"""
    id: str = Field(..., description="Unique identifier")
    space_id: str = Field(..., description="Space this flashcard belongs to")
    question: str = Field(..., description="Front of the card (question/prompt)")
    answer: str = Field(..., description="Back of the card (answer/explanation)")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    topic: Optional[str] = Field(None, description="Topic matching the concept covered")
    mastery: MasteryLevel = Field(default=MasteryLevel.NEW)
    source_type: str = Field(default="manual", description="Source: manual, generated, notebook")
    source_id: Optional[str] = Field(None, description="Source ID (e.g., notebook entry ID)")
    tags: List[str] = Field(default_factory=list)
    review_count: int = Field(default=0, description="Number of times reviewed")
    correct_count: int = Field(default=0, description="Number of times answered correctly")
    
    # SM-2 Spaced Repetition Fields
    interval: int = Field(default=0, description="Interval in days before next review")
    repetitions: int = Field(default=0, description="Number of successful consecutive reviews")
    ease_factor: float = Field(default=2.5, description="SM-2 Ease factor")
    
    last_reviewed: Optional[datetime] = None
    next_review: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FlashcardCreate(BaseModel):
    """Request model for creating a flashcard"""
    space_id: str
    question: str
    answer: str
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    source_type: str = "manual"
    source_id: Optional[str] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class FlashcardUpdate(BaseModel):
    """Request model for updating a flashcard"""
    question: Optional[str] = None
    answer: Optional[str] = None
    difficulty: Optional[DifficultyLevel] = None
    mastery: Optional[MasteryLevel] = None
    tags: Optional[List[str]] = None


class FlashcardReview(BaseModel):
    """Request model for reviewing a flashcard"""
    quality: int = Field(..., description="0-5 score user gives after seeing answer (0-2 = forgot, 3 = hard, 4 = good, 5 = easy)")


class FlashcardGenerateRequest(BaseModel):
    """Request to generate flashcards from content"""
    space_id: str
    source_type: str = Field(..., description="Source type: notebook, file, text")
    source_ids: Optional[List[str]] = Field(None, description="IDs of notebook entries or files")
    text_content: Optional[str] = Field(None, description="Direct text content to generate from")
    num_cards: int = Field(default=5, description="Number of flashcards to generate")
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM


# ============================================================================
# QUIZ MODELS
# ============================================================================

class QuestionType(str, Enum):
    """Type of quiz question"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"


class QuizQuestion(BaseModel):
    """A single quiz question"""
    id: str = Field(..., description="Unique identifier")
    question: str = Field(..., description="Question text")
    type: QuestionType = Field(..., description="Question type")
    options: Optional[List[str]] = Field(None, description="Options for multiple choice")
    correct_answer: str = Field(..., description="Correct answer")
    explanation: Optional[str] = Field(None, description="Explanation of the answer")
    points: int = Field(default=1, description="Points for this question")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)


class Quiz(BaseModel):
    """A quiz session"""
    id: str = Field(..., description="Unique identifier")
    space_id: str = Field(..., description="Space this quiz belongs to")
    title: str = Field(..., description="Quiz title")
    description: Optional[str] = None
    questions: List[QuizQuestion] = Field(..., description="List of questions")
    source_type: str = Field(default="manual", description="Source: manual, generated, notebook, file")
    source_ids: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuizCreate(BaseModel):
    """Request model for creating a quiz"""
    space_id: str
    title: str
    description: Optional[str] = None
    questions: List[QuizQuestion] = []
    source_type: str = "manual"
    source_ids: Optional[List[str]] = None


class QuizGenerateRequest(BaseModel):
    """Request to generate a quiz from content"""
    space_id: str
    title: str
    source_type: str = Field(..., description="Source type: notebook, file, text")
    source_ids: Optional[List[str]] = Field(None, description="IDs of notebook entries or files")
    text_content: Optional[str] = Field(None, description="Direct text content")
    num_questions: int = Field(default=5, description="Number of questions")
    question_types: List[QuestionType] = Field(
        default=[QuestionType.MULTIPLE_CHOICE],
        description="Types of questions to include"
    )
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM


class QuizAnswer(BaseModel):
    """User's answer to a quiz question"""
    question_id: str
    answer: str
    time_spent: Optional[int] = Field(None, description="Time spent in seconds")


class QuizSubmission(BaseModel):
    """User's quiz submission"""
    quiz_id: str
    answers: List[QuizAnswer]


class QuizResult(BaseModel):
    """Result of a quiz submission"""
    quiz_id: str
    submission_id: str = Field(..., description="Unique submission ID")
    total_questions: int
    correct_answers: int
    incorrect_answers: int
    score_percentage: float
    total_points: int
    earned_points: int
    answers: List[Dict[str, Any]] = Field(..., description="Detailed answer results")
    completed_at: datetime = Field(default_factory=datetime.now)
    time_taken: Optional[int] = Field(None, description="Total time in seconds")


class QuizHistory(BaseModel):
    """Quiz attempt history"""
    quiz_id: str
    space_id: str
    quiz_title: str
    results: List[QuizResult] = Field(default_factory=list)
    best_score: float = Field(default=0.0)
    average_score: float = Field(default=0.0)
    attempts_count: int = Field(default=0)
