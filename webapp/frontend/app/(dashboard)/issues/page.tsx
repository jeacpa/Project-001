import { Typography } from "@mui/material";
import { PageContainer } from "@toolpad/core";
import IssuesList from "../IssuesList";

export default function SearchPage() {
  return (    
    <PageContainer breadcrumbs={[]}>
      <Typography>
      <div>
		The issues here are taken from our &nbsp;
		<a href="https://github.com/jeacpa/Project-001">
			github repository page
		</a>
		&nbsp; and displayed here. We use github to maintain our code.
		<br />
		Issues are located &nbsp;
		<a href="https://github.com/jeacpa/Project-001/issues">
		here
		</a>

</div>
        The issues here are taking from our github repository page and displayed here.  We need help writing the code to grab
        the issue from https://github.com/jeacpa/Project-001/issues.
      </Typography>
      <IssuesList />
    </PageContainer>
  );
}
