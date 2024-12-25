import os
import sys
from pathlib import Path
import fnmatch
from typing import List, Set

class GitignoreParser:
    """
    Handles parsing and matching of .gitignore patterns.
    """
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.ignore_patterns: List[str] = []
        self.load_gitignore()

    def load_gitignore(self) -> None:
        """
        Loads all .gitignore files from the root directory and its parents.
        This respects the git behavior of considering all .gitignore files
        in the hierarchy.
        """
        current_dir = self.root_dir
        while current_dir.exists():
            gitignore_path = current_dir / '.gitignore'
            if gitignore_path.is_file():
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    # Filter out empty lines and comments
                    patterns = [
                        line.strip() 
                        for line in f 
                        if line.strip() and not line.startswith('#')
                    ]
                    self.ignore_patterns.extend(patterns)
            # Move up to parent directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir

    def should_ignore(self, path: Path) -> bool:
        """
        Determines if a path should be ignored based on .gitignore rules.
        
        Args:
            path: Path to check against gitignore rules
            
        Returns:
            bool: True if the path should be ignored, False otherwise
        """
        # Get relative path from root directory
        rel_path = str(path.relative_to(self.root_dir))
        
        # Convert Windows paths to forward slashes for consistency
        rel_path = rel_path.replace(os.sep, '/')
        
        for pattern in self.ignore_patterns:
            # Handle both directory and file patterns
            if pattern.endswith('/'):
                if fnmatch.fnmatch(f"{rel_path}/", pattern):
                    return True
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False

def concat_py_files(root_dir: str) -> None:
    """
    Concatenates JavaScript files respecting directory structure and .gitignore rules.
    Creates separate concatenated files for each directory level.
    
    Args:
        root_dir: Root directory to start processing from
    """
    root_path = Path(root_dir).resolve()
    output_dir = root_path / 'concated'
    output_dir.mkdir(exist_ok=True)
    
    # Initialize gitignore parser
    gitignore = GitignoreParser(root_path)
    
    def process_directory(dir_path: Path, relative_output_path: Path) -> None:
        """
        Recursively processes directories and their JavaScript files.
        
        Args:
            dir_path: Current directory being processed
            relative_output_path: Relative path for maintaining directory structure
        """
        # Skip if directory should be ignored
        if gitignore.should_ignore(dir_path):
            print(f'Skipping ignored directory: {dir_path}')
            return
        
        # Create corresponding output directory
        current_output_dir = output_dir / relative_output_path
        current_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get all py files in current directory
        py_files = [
            f for f in dir_path.glob('*.py')
            if not gitignore.should_ignore(f)
        ]
        
        if py_files:
            # Create concatenated file for current directory
            concatenated = []
            for py_file in py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        full_path = py_file.resolve()
                        concatenated.append(
                            f'// Source: {full_path}\n'
                            f'// Relative path: {py_file.relative_to(root_path)}\n'
                            f'{f.read()}\n'
                        )
                except Exception as e:
                    print(f'Error reading {py_file}: {str(e)}')
                    continue
            
            # Write concatenated file for current directory
            output_file = current_output_dir / f'{dir_path.name}_concatenated.py'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(concatenated))
            
            print(f'Created {output_file} from {len(py_files)} files')
        
        # Process subdirectories
        for subdir in dir_path.iterdir():
            if subdir.is_dir() and subdir.name != 'concated':
                process_directory(
                    subdir,
                    relative_output_path / subdir.name
                )

    # Start processing from root directory
    process_directory(root_path, Path(''))

if __name__ == '__main__':
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    try:
        concat_py_files(target_dir)
        print('Concatenation complete!')
    except Exception as e:
        print(f'Error: {str(e)}')
        sys.exit(1)