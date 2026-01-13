import Logo from '../common/Logo';

export default function Footer() {
    const links = [
        { label: 'X', href: 'https://x.com/mmoment_xyz', icon: 'twitter' },
        { label: 'GitHub', href: 'https://github.com/0xmigi/mmoment', icon: 'github' },
        { label: 'Blog', href: 'https://paragraph.com/@0xmigi.eth/', icon: 'blog' },
    ];

    return (
        <footer className="bg-black text-white pt-4 pb-12">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex flex-col md:flex-row items-center justify-between gap-8">
                    {/* Logo & Brand */}
                    <div className="flex items-center gap-3">
                        <Logo width={28} height={20} className="text-white" />
                        <span className="text-lg font-semibold">Moment</span>
                    </div>

                    {/* Icon Links */}
                    <div className="flex items-center">
                        {links.map((link) => (
                            <a
                                key={link.label}
                                href={link.href}
                                target="_blank"
                                rel="noopener noreferrer"
                                aria-label={link.label}
                                className="text-gray-400 hover:text-white transition-colors p-2"
                            >
                                {link.icon === 'twitter' && (
                                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                                    </svg>
                                )}
                                {link.icon === 'github' && (
                                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                        <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
                                    </svg>
                                )}
                                {link.icon === 'blog' && (
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                                    </svg>
                                )}
                            </a>
                        ))}
                    </div>
                </div>

                {/* Bottom */}
                <div className="mt-8 pt-8 border-t border-gray-800 text-center text-gray-500 text-xs">
                    © {new Date().getFullYear()} Moment. Secured by Solana.
                </div>
            </div>
        </footer>
    );
}
