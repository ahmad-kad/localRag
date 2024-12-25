def test_process_directory(sample_codebase, indexer):
    """Test processing a directory of code files."""
    docs = indexer.process_directory(sample_codebase)
    assert len(docs) > 0
    assert any("add" in doc.content for doc in docs)
    assert any("process_data" in doc.content for doc in docs)

def test_create_index(indexer):
    """Test index creation."""
    assert indexer.index is not None
    assert indexer.index.storage_context is not None

def test_save_load_index(indexer, tmp_path):
    """Test saving and loading an index."""
    save_path = tmp_path / "test_index"
    indexer.save_index(save_path)
    assert save_path.exists()
    
    new_indexer = CodeIndexer()
    new_indexer.load_index(save_path)
    assert new_indexer.index is not None

def test_process_directory(sample_codebase, indexer):
    """Test processing a directory of code files."""
    docs = indexer.process_directory(sample_codebase)
    assert len(docs) > 0
    assert any("add" in doc.content for doc in docs)
    assert any("process_data" in doc.content for doc in docs)

def test_create_index(indexer):
    """Test index creation."""
    assert indexer.index is not None
    assert indexer.index.storage_context is not None

def test_save_load_index(indexer, tmp_path):
    """Test saving and loading an index."""
    save_path = tmp_path / "test_index"
    indexer.save_index(save_path)
    assert save_path.exists()
    
    new_indexer = CodeIndexer()
    new_indexer.load_index(save_path)
    assert new_indexer.index is not None