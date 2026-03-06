import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    handleRetry = () => {
        this.setState({ hasError: false, error: null });
    };

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '60vh',
                    padding: '2rem',
                    textAlign: 'center',
                }}>
                    <div style={{
                        fontSize: '3rem',
                        marginBottom: '1rem',
                    }}>⚠️</div>
                    <h2 style={{
                        fontSize: '1.5rem',
                        marginBottom: '0.5rem',
                        color: '#333',
                    }}>Something went wrong</h2>
                    <p style={{
                        color: '#666',
                        marginBottom: '1.5rem',
                        maxWidth: '400px',
                    }}>
                        An unexpected error occurred. Please try again.
                    </p>
                    <button
                        onClick={this.handleRetry}
                        style={{
                            padding: '0.75rem 1.5rem',
                            backgroundColor: '#007bff',
                            color: 'white',
                            border: 'none',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            fontSize: '1rem',
                            fontWeight: '600',
                        }}
                    >
                        Try Again
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
