# Contributing to Stock Market Analytics Platform

Thank you for your interest in contributing! Here's how to get started.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stock-dashboard.git
   cd stock-dashboard
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate  # Windows
   source venv/bin/activate  # Mac/Linux
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Making Changes

1. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and test thoroughly

3. **Follow code style**:
   - Use PEP 8 style guide
   - Write descriptive docstrings
   - Add comments for complex logic

4. **Test your code**:
   ```bash
   # Run the app locally
   streamlit run forecasting_app.py
   
   # Test the Jupyter notebook
   jupyter notebook Stock_Market_EDA_Forecasting.ipynb
   ```

## Committing Changes

- Write clear, descriptive commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issues when applicable (#123)

Example:
```bash
git commit -m "Add ARIMA forecast confidence intervals (#42)"
```

## Submitting a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to related issues
   - Screenshot/example if UI changes
   - Updated documentation if needed

3. **Address feedback** from reviewers

## Areas for Contribution

### High Priority
- [ ] Additional forecasting models (Prophet, GARCH, VAR)
- [ ] Real-time data streaming
- [ ] Portfolio optimization algorithms
- [ ] Unit tests and integration tests

### Medium Priority
- [ ] Options pricing models
- [ ] Value at Risk (VaR) calculations
- [ ] More technical indicators
- [ ] Database integration
- [ ] User authentication

### Low Priority
- [ ] UI/UX improvements
- [ ] Documentation updates
- [ ] Performance optimizations
- [ ] Additional visualizations

## Bug Reports

Found a bug? Please open an issue with:
1. **Title**: Brief description of the bug
2. **Description**: Detailed explanation
3. **Steps to Reproduce**: How to trigger the bug
4. **Expected vs Actual**: What should happen vs what happens
5. **Environment**: OS, Python version, package versions

## Feature Requests

Have an idea? Open an issue with:
1. **Use Case**: Why is this feature needed?
2. **Solution**: How should it work?
3. **Examples**: Mock-ups or example code (optional)

## Documentation

- Update relevant `.md` files when changing functionality
- Keep README.md in sync with code
- Document all new classes/functions with docstrings
- Include usage examples for major features

## Code Review Process

- At least one approval required before merge
- All CI/CD tests must pass
- No merge conflicts
- Documentation must be updated

## Questions?

- Check existing issues and discussions
- Read documentation in README.md
- Review code comments and docstrings

---

**Thank you for contributing! 🎉**
