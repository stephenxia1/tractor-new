import RoomClient from './RoomClient';

export default function RoomPage({ params }: { params: { code: string } }) {
  return <RoomClient code={params.code.toUpperCase()} />;
}
