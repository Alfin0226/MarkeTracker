import React, { useState, useEffect } from 'react';
import { fetchTransactions } from '../utils/api';
import '../styles/Portfolio.css';

function TransactionHistory() {
    const [transactions, setTransactions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);

    const loadTransactions = async (pg = 1) => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchTransactions(pg, 25);
            setTransactions(data.transactions);
            setTotalPages(data.pages);
            setPage(data.current_page);
        } catch (err) {
            setError('Failed to load transactions');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadTransactions();
    }, []);

    const formatDate = (isoString) => {
        if (!isoString) return 'N/A';
        const date = new Date(isoString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit'
        });
    };

    return (
        <div className="portfolio-container">
            <h2>Transaction History</h2>

            {error && <div className="alert alert-danger">{error}</div>}

            {loading ? (
                <div className="holdings-section">
                    <div className="loading-skeleton" style={{ height: '300px' }}></div>
                </div>
            ) : transactions.length === 0 ? (
                <div className="holdings-section" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                    <p style={{ fontSize: '1.2rem' }}>No transactions yet</p>
                    <p>Execute a trade from your Portfolio page to see history here.</p>
                </div>
            ) : (
                <>
                    <div className="holdings-section">
                        <table className="table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Action</th>
                                    <th>Symbol</th>
                                    <th>Shares</th>
                                    <th>Price</th>
                                    <th>Total</th>
                                </tr>
                            </thead>
                            <tbody>
                                {transactions.map((t) => (
                                    <tr key={t.id}>
                                        <td style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                            {formatDate(t.timestamp)}
                                        </td>
                                        <td>
                                            <span style={{
                                                padding: '0.25rem 0.6rem',
                                                borderRadius: 'var(--radius-sm)',
                                                fontSize: '0.8rem',
                                                fontWeight: 700,
                                                textTransform: 'uppercase',
                                                letterSpacing: '0.5px',
                                                background: t.action === 'buy' ? 'var(--color-success-bg)' : 'var(--color-danger-bg)',
                                                color: t.action === 'buy' ? 'var(--color-success)' : 'var(--color-danger)',
                                                border: `1px solid ${t.action === 'buy' ? 'rgba(0,214,143,0.3)' : 'rgba(255,107,107,0.3)'}`,
                                            }}>
                                                {t.action}
                                            </span>
                                        </td>
                                        <td style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>{t.symbol}</td>
                                        <td>{t.shares}</td>
                                        <td>${t.price.toFixed(2)}</td>
                                        <td style={{ fontWeight: 600 }}>
                                            ${t.total.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {totalPages > 1 && (
                        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', marginTop: '1.5rem' }}>
                            <button
                                className="portfolio-refresh-btn"
                                onClick={() => loadTransactions(page - 1)}
                                disabled={page <= 1}
                            >
                                ← Previous
                            </button>
                            <span style={{ padding: '0.7rem 1rem', color: 'var(--text-secondary)' }}>
                                Page {page} of {totalPages}
                            </span>
                            <button
                                className="portfolio-refresh-btn"
                                onClick={() => loadTransactions(page + 1)}
                                disabled={page >= totalPages}
                            >
                                Next →
                            </button>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

export default TransactionHistory;
