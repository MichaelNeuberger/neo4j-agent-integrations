import type { Metadata } from "next";
import { Provider } from "@/components/Provider";
import "./globals.css";

export const metadata: Metadata = {
  title: "Healthcare Context Graph",
  description: "Patient care, clinical encounters, diagnoses, treatments, and provider networks",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
