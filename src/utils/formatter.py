"""Beautiful output formatting with colors and styling."""

import re
from typing import Optional


class OutputFormatter:
    """
    Formats output with colors, bold text, and visual enhancements.
    Makes CLI experience delightful and professional.
    """
    
    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
        
        # Foreground colors
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        
        # Bright foreground
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
    }
    
    @classmethod
    def format(cls, text: str, enable_colors: bool = True) -> str:
        """
        Format text with markdown-style syntax and colors.
        Converts **bold**, *italic*, and adds colors to special elements.
        """
        if not enable_colors:
            # Strip markdown for plain text
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            text = re.sub(r'\*(.*?)\*', r'\1', text)
            return text
        
        # Convert **bold** to ANSI bold
        text = re.sub(r'\*\*(.*?)\*\*', f"{cls.COLORS['bold']}\\1{cls.COLORS['reset']}", text)
        
        # Convert *italic* to ANSI italic
        text = re.sub(r'\*(.*?)\*', f"{cls.COLORS['italic']}\\1{cls.COLORS['reset']}", text)
        
        # Colorize special markers
        text = text.replace('✅', f"{cls.COLORS['bright_green']}✅{cls.COLORS['reset']}")
        text = text.replace('❌', f"{cls.COLORS['bright_red']}❌{cls.COLORS['reset']}")
        text = text.replace('⚠️', f"{cls.COLORS['bright_yellow']}⚠️{cls.COLORS['reset']}")
        text = text.replace('📊', f"{cls.COLORS['bright_cyan']}📊{cls.COLORS['reset']}")
        text = text.replace('💡', f"{cls.COLORS['bright_yellow']}💡{cls.COLORS['reset']}")
        text = text.replace('📈', f"{cls.COLORS['bright_blue']}📈{cls.COLORS['reset']}")
        text = text.replace('💰', f"{cls.COLORS['bright_yellow']}💰{cls.COLORS['reset']}")
        text = text.replace('⏱️', f"{cls.COLORS['cyan']}⏱️{cls.COLORS['reset']}")
        text = text.replace('🤖', f"{cls.COLORS['bright_magenta']}🤖{cls.COLORS['reset']}")
        text = text.replace('💾', f"{cls.COLORS['bright_blue']}💾{cls.COLORS['reset']}")
        text = text.replace('📭', f"{cls.COLORS['dim']}📭{cls.COLORS['reset']}")
        
        # Colorize headers
        lines = text.split('\n')
        formatted_lines = []
        for line in lines:
            # Key Findings header
            if line.strip().startswith('📈 Key Findings:'):
                line = f"{cls.COLORS['bold']}{cls.COLORS['bright_blue']}{line}{cls.COLORS['reset']}"
            # Insights header
            elif line.strip().startswith('💡 Insights:'):
                line = f"{cls.COLORS['bold']}{cls.COLORS['bright_yellow']}{line}{cls.COLORS['reset']}"
            # Results header
            elif 'Results' in line and '📊' in line:
                line = f"{cls.COLORS['bold']}{cls.COLORS['bright_cyan']}{line}{cls.COLORS['reset']}"
            # Bullet points
            elif line.strip().startswith('•'):
                line = f"{cls.COLORS['cyan']}  •{cls.COLORS['reset']}{line.strip()[1:]}"
            elif line.strip().startswith('-'):
                line = f"{cls.COLORS['cyan']}  -{cls.COLORS['reset']}{line.strip()[1:]}"
            
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    @classmethod
    def success(cls, message: str) -> str:
        """Format success message."""
        return f"{cls.COLORS['bright_green']}✅ {message}{cls.COLORS['reset']}"
    
    @classmethod
    def error(cls, message: str) -> str:
        """Format error message."""
        return f"{cls.COLORS['bright_red']}❌ {message}{cls.COLORS['reset']}"
    
    @classmethod
    def warning(cls, message: str) -> str:
        """Format warning message."""
        return f"{cls.COLORS['bright_yellow']}⚠️  {message}{cls.COLORS['reset']}"
    
    @classmethod
    def info(cls, message: str) -> str:
        """Format info message."""
        return f"{cls.COLORS['bright_cyan']}ℹ️  {message}{cls.COLORS['reset']}"
    
    @classmethod
    def highlight(cls, text: str) -> str:
        """Highlight important text."""
        return f"{cls.COLORS['bold']}{cls.COLORS['bright_yellow']}{text}{cls.COLORS['reset']}"

