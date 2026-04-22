import React, { useState, useEffect } from 'react';
import { fetchWatchlist, removeFromWatchlist } from '../utils/api';
import { useNavigate } from 'react-router-dom';
import '../styles/Portfolio.css';

function Watchlist() {
    const [watchlist, setWatchlist] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const loadWatchlist = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchWatchlist();
            setWatchlist(data.watchlist);
        } catch (err) {
            setError('Failed to load watchlist');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadWatchlist();
    }, []);

    const handleRemove = async (symbol) => {
        try {
            await removeFromWatchlist(symbol);
            setWatchlist(prev => prev.filter(item => item.symbol !== symbol));
        } catch (err) {
            setError('Failed to remove from watchlist');
        }
    };

    const formatDate = (isoString) => {
        if (!isoString) return 'N/A';
        return new Date(isoString).toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric'
        });
    };

    return (
        <div className="portfolio-container">
            <h2>Watchlist</h2>

            {error && <div className="alert alert-danger">{error}</div>}

            {loading ? (
                <div className="portfolio-summary">
                    {[1, 2, 3].map(i => (
                        <div className="summary-card" key={i}>
                            <div className="loading-skeleton" style={{ height: '80px' }}></div>
                        </div>
                    ))}
                </div>
            ) : watchlist.length === 0 ? (
                <div className="holdings-section" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                    <p style={{ fontSize: '3rem', marginBottom: '1rem' }}>👀</p>
                    <p style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>Your watchlist is empty</p>
                    <p>Add stocks from the Dashboard to keep an eye on them.</p>
                </div>
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1rem' }}>
                    {watchlist.map((item) => (
                        <div key={item.symbol} className="summary-card" style={{ cursor: 'pointer', position: 'relative' }}>
                            <div onClick={() => navigate(`/dashboard/${item.symbol}`)}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                                    <span style={{ fontWeight: 700, fontSize: '1.2rem', color: 'var(--accent-primary)' }}>
                                        {item.symbol}
                                    </span>
                                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                        Added {formatDate(item.added_at)}
                                    </span>
                                </div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-heading)' }}>
                                    {item.current_price !== null
                                        ? `$${item.current_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                                        : 'Price unavailable'
                                    }
                                </div>
                            </div>
                            <button
                                onClick={(e) => { e.stopPropagation(); handleRemove(item.symbol); }}
                                style={{
                                    position: 'absolute', top: '0.75rem', right: '0.75rem',
                                    background: 'none', border: 'none', color: 'var(--text-muted)',
                                    cursor: 'pointer', fontSize: '1rem', padding: '0.25rem',
                                    transition: 'color 0.15s ease',
                                }}
                                onMouseOver={(e) => e.currentTarget.style.color = 'var(--color-danger)'}
                                onMouseOut={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
                                title="Remove from watchlist"
                            >
                                ✕
                            </button>
                        </div>
                    ))}
                </div>
            )}

            <button className="portfolio-refresh-btn" onClick={loadWatchlist} disabled={loading} style={{ marginTop: '1.5rem' }}>
                {loading ? 'Loading...' : '↻ Refresh'}
            </button>
        </div>
    );
}

export default Watchlist;
