export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <head>
        <script src="/vendor/face_mesh/face_mesh.js"></script>
        <script src="/vendor/camera_utils/camera_utils.js"></script>
      </head>
      <body>{children}</body>
    </html>
  );
}
